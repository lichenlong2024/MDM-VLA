"""
deploy.py

Starts VLA server which the client can query to get robot actions.
"""
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Union
import time
import sys

import draccus
import msgpack
import torch
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from PIL import Image
import msgpack_numpy


# Append project root to sys.path
sys.path.append("../..")

from experiments.robot.openvla_utils import (
    find_checkpoint_file,
    get_action_head,
    get_processor,
    get_proprio_projector,
    load_component_state_dict,
)
from experiments.robot.robot_utils import (
    get_action,
    get_action_with_action_id,
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)
from prismatic.models.action_heads import StageMoEActionHead
from prismatic.models.action_id_discriminator import ActionIDDiscriminator
from prismatic.models.backbones.vision.dinosiglip_vit import DinoSigLIPViTBackbone


# Set up logging to display timestamp, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


@dataclass
class DeployConfig:
    # fmt: off

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port
    device: str = "cuda:0"                                              # Device to run model on

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_minivlm: bool = True                         # If True, uses minivlm
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 3                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 25                    # Number of actions to execute open-loop before requerying policy
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 42                                   # Random Seed (for reproducibility)

    # fmt: on
    save_version: str = "vla-adapter"                # version of 
    phase: str = "Inference"
    use_pro_version: bool = True
    use_moe_action_head: bool = False
    use_action_id_discriminator: bool = False
    stage_definitions: str = "0:0-2,1:3-10,2:11-17"
    num_action_ids: int = 18


def _parse_stage_definitions(stage_definitions_str: str) -> Dict[int, list[int]]:
    stage_definitions = {}
    for stage_def in stage_definitions_str.split(","):
        stage_id, action_range = stage_def.split(":")
        start, end = map(int, action_range.split("-"))
        stage_definitions[int(stage_id)] = list(range(start, end + 1))
    return stage_definitions


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensors or nested tensor lists to a CPU numpy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    if isinstance(value, (list, tuple)):
        return np.asarray([_to_numpy(item) for item in value], dtype=np.float32)
    return np.asarray(value, dtype=np.float32)



def initialize_model(cfg: DeployConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)
    model.set_version(cfg.save_version)
    
    # Get number of vision patches
    NUM_PATCHES = model.vision_backbone.get_num_patches() * model.vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings
    if cfg.use_proprio:
        NUM_PATCHES += 1
    cfg.num_task_tokens=NUM_PATCHES

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=14,  # 14-dimensional proprio for aloha
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression:
        if cfg.use_moe_action_head:
            action_head = StageMoEActionHead(
                input_dim=model.llm_dim,
                hidden_dim=model.llm_dim,
                action_dim=7,
                use_pro_version=cfg.use_pro_version,
                stage_definitions=_parse_stage_definitions(cfg.stage_definitions),
                num_action_ids=cfg.num_action_ids,
            )
            checkpoint_path = find_checkpoint_file(str(cfg.pretrained_checkpoint), "action_head")
            action_head.load_state_dict(load_component_state_dict(checkpoint_path))
            action_head = action_head.to(torch.bfloat16).to(cfg.device)
            action_head.eval()
        else:
            action_head = get_action_head(cfg, model.llm_dim)

    action_id_discriminator = None
    if cfg.use_action_id_discriminator:
        vision_backbone = DinoSigLIPViTBackbone(
            vision_backbone_id="dinosiglip-vit-so-224px",
            image_resize_strategy="resize-naive",
        )
        action_id_discriminator = ActionIDDiscriminator(
            num_action_ids=cfg.num_action_ids,
            vision_backbone=vision_backbone,
            proprio_dim=14,
            llm_dim=model.llm_dim,
        )
        checkpoint_path = find_checkpoint_file(str(cfg.pretrained_checkpoint), "action_id_discriminator")
        action_id_discriminator.load_state_dict(load_component_state_dict(checkpoint_path))
        action_id_discriminator = action_id_discriminator.to(torch.bfloat16).to(cfg.device)
        action_id_discriminator.eval()

    # Get OpenVLA processor
    processor = get_processor(cfg)


    return model, processor, action_head, proprio_projector, action_id_discriminator


class MsgPackResponse(Response):
    """Custom FastAPI Response class to automatically encode response data into MessagePack."""

    media_type = "application/msgpack"

    def render(self, content: Any) -> bytes:
        return msgpack.packb(content, default=msgpack_numpy.encode, use_bin_type=True)


# === Server Interface ===
class VLAServer:
    def __init__(self, cfg: DeployConfig):
        """
        A simple server for VLA models, exposing `/act` endpoint.
        This server receives observations and instructions via MessagePack,
        and returns predicted actions in MessagePack format.
        """
        self.cfg = cfg
        (
            self.model,
            self.processor,
            self.action_head,
            self.proprio_projector,
            self.action_id_discriminator,
        ) = initialize_model(cfg)
        self.resize_size = get_image_resize_size(cfg)
        set_seed_everywhere(self.cfg.seed)
        self.app = FastAPI()

        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            """
            Middleware to log request details including processing time.
            """
            start_time = time.time()
            response = await call_next(request)
            process_time = (time.time() - start_time) * 1000  # in milliseconds
            logging.info(f'"{request.method} {request.url.path}" {response.status_code} - {process_time:.2f}ms')
            return response

        self.app.post("/act", response_class=MsgPackResponse)(self.get_server_action)

    async def get_server_action(self, request: Request) -> Dict[str, Any]:
        """Handles a single action prediction request using MessagePack."""
        if request.headers.get("content-type") != "application/msgpack":
            raise HTTPException(
                status_code=415, detail="Unsupported Media Type. 'application/msgpack' is required."
            )
        try:
            body = await request.body()
            batch = msgpack.unpackb(body, object_hook=msgpack_numpy.decode, raw=False)
            
            # Extract unnorm_key and instruction from the batch
            unnorm_key = batch.pop("unnorm_key")
            instruction = batch.pop("instruction")

            # Update cfg with the unnorm_key from the client
            self.cfg.unnorm_key = unnorm_key

            action_id_pred = None
            if self.cfg.use_moe_action_head or self.cfg.use_action_id_discriminator:
                actions, action_id_pred = get_action_with_action_id(
                    self.cfg,
                    self.model,
                    batch,
                    instruction,
                    processor=self.processor,
                    action_head=self.action_head,
                    proprio_projector=self.proprio_projector,
                    action_id_discriminator=self.action_id_discriminator,
                    use_moe_action_head=self.cfg.use_moe_action_head,
                )
            else:
                actions = get_action(
                    self.cfg,
                    self.model,
                    batch,
                    instruction,
                    processor=self.processor,
                    action_head=self.action_head,
                    proprio_projector=self.proprio_projector,
                )
            
            response = {"actions": _to_numpy(actions).tolist()}
            if action_id_pred is not None:
                response["action_id_pred"] = int(_to_numpy(action_id_pred).reshape(-1)[0])
            return response

        except msgpack.UnpackException:
            raise HTTPException(status_code=400, detail="Invalid MessagePack data provided.")
        except Exception:
            logging.error(traceback.format_exc())
            # Re-raise as a generic 500 error to avoid leaking implementation details.
            raise HTTPException(status_code=500, detail="An internal server error occurred.")

    def run(self) -> None:
        """Starts the Uvicorn server."""
        uvicorn.run(self.app, host=self.cfg.host, port=self.cfg.port, access_log=False, timeout_keep_alive=120)


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = VLAServer(cfg)
    server.run()


if __name__ == "__main__":
    deploy()
