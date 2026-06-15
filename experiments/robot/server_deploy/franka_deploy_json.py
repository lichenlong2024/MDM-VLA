"""franka_deploy_json.py

A lightweight JSON+Base64 inference server for Franka real robot.

This intentionally mirrors the observation/action semantics used in the LIBERO eval pipeline
(`experiments/robot/libero/run_libero_eval.py`) but exposes an HTTP JSON endpoint that is easy
for a ROS+MoveIt client to call.

Endpoint:
  POST /act
Request JSON:
  {
    "full_image": "<base64 png/jpg>",
    "wrist_image": "<base64 png/jpg>",
    "goal": "...",
    "state": [x,y,z,qx,qy,qz,qw,gripper]
  }
Response JSON:
  {"action": [[dx,dy,dz,dr,dp,dyaw,gripper]]}

Notes:
- The server returns only the FIRST action (shape [1,7]) since you train with L1 loss on the
  first predicted action.
- Image preprocessing (resize/crop) is delegated to the same utilities used in simulation.
"""

import base64
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Project root (…/VLA-Adapter), works regardless of cwd
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import draccus
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request

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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


@dataclass
class FrankaDeployConfig:
    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    device: str = "cuda:0"

    # Model
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""

    use_l1_regression: bool = True
    use_minivlm: bool = True
    use_film: bool = False

    # Real robot uses 2 images (full + wrist)
    num_images_in_input: int = 2

    # Proprio/state is 8D (pos3 + quat4 + gripper1)
    use_proprio: bool = True
    proprio_dim: int = 8

    center_crop: bool = True

    # Return only action[0], but internally we can ask for a chunk.
    num_open_loop_steps: int = 8

    # Optional override. If empty, auto-detected from checkpoint/dataset_statistics.json.
    unnorm_key: str = ""

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    seed: int = 7

    save_version: str = "vla-adapter"
    phase: str = "Inference"
    use_pro_version: bool = True

    # Action-ID routed MoE inference. Keep both False for the baseline path.
    use_moe_action_head: bool = False
    use_action_id_discriminator: bool = False
    stage_definitions: str = "0:0-2,1:3-10,2:11-17"
    num_action_ids: int = 18


def _parse_stage_definitions(stage_definitions_str: str) -> Dict[int, List[int]]:
    stage_definitions = {}
    for stage_def in stage_definitions_str.split(","):
        stage_id, action_range = stage_def.split(":")
        start, end = map(int, action_range.split("-"))
        stage_definitions[int(stage_id)] = list(range(start, end + 1))
    return stage_definitions


def _maybe_to_device(cfg: FrankaDeployConfig, model: torch.nn.Module) -> torch.nn.Module:
    # get_vla() inside get_model() already uses the global DEVICE in experiments/robot/robot_utils.py,
    # but we still set torch device context here for clarity.
    return model


def _dataset_statistics_path(checkpoint: Union[str, Path]) -> Path:
    """Return dataset_statistics.json path under the checkpoint directory."""
    checkpoint_dir = Path(checkpoint).expanduser().resolve()
    return checkpoint_dir / "dataset_statistics.json"


def _load_norm_stats_from_checkpoint(checkpoint: Union[str, Path]) -> Dict[str, Any]:
    """Load action/proprio normalization stats bundled with the checkpoint."""
    stats_path = _dataset_statistics_path(checkpoint)
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"未找到 dataset_statistics.json: {stats_path}\n"
            f"请确认 --pretrained_checkpoint 指向包含该文件的 checkpoint 目录。"
        )
    with open(stats_path, "r", encoding="utf-8") as f:
        norm_stats = json.load(f)
    if not norm_stats:
        raise ValueError(f"dataset_statistics.json 为空: {stats_path}")
    return norm_stats


def _select_unnorm_key(
    available_keys: List[str],
    checkpoint: Union[str, Path],
    user_key: str = "",
) -> str:
    """Pick unnorm_key: explicit CLI > path name match > sole key > first key."""
    if user_key and user_key.strip():
        key = user_key.strip()
        if key not in available_keys:
            raise ValueError(
                f"指定的 unnorm_key '{key}' 不在 dataset_statistics.json 中。"
                f" 可用键: {available_keys}"
            )
        return key

    if len(available_keys) == 1:
        return available_keys[0]

    checkpoint_lower = str(Path(checkpoint).expanduser()).lower()
    for key in available_keys:
        if key.lower() in checkpoint_lower:
            logging.info(f"根据 checkpoint 路径匹配 unnorm_key: '{key}'")
            return key

    selected = available_keys[0]
    logging.warning(
        f"dataset_statistics.json 含多个键 {available_keys}，"
        f"未在路径中匹配到名称，使用第一个: '{selected}'"
    )
    return selected


def _resolve_unnorm_key(cfg: FrankaDeployConfig, model) -> Tuple[str, Path]:
    """Resolve unnorm_key from checkpoint stats (no manual --unnorm_key required)."""
    if not cfg.pretrained_checkpoint:
        raise ValueError("--pretrained_checkpoint 必须指定 checkpoint 目录路径。")

    stats_path = _dataset_statistics_path(cfg.pretrained_checkpoint)
    norm_stats = _load_norm_stats_from_checkpoint(cfg.pretrained_checkpoint)
    available_keys = list(norm_stats.keys())

    selected_key = _select_unnorm_key(available_keys, cfg.pretrained_checkpoint, cfg.unnorm_key)

    if hasattr(model, "norm_stats") and model.norm_stats:
        model_keys = list(model.norm_stats.keys())
        if selected_key not in model_keys:
            raise ValueError(
                f"模型 norm_stats 键 {model_keys} 与 dataset_statistics.json 键 "
                f"{available_keys} 不一致，选中键 '{selected_key}' 不可用。"
            )
    else:
        model.norm_stats = norm_stats

    logging.info(
        f"已从 {stats_path} 加载归一化统计，unnorm_key='{selected_key}'，"
        f"可用键: {available_keys}"
    )
    return selected_key, stats_path


def _auto_detect_unnorm_key(cfg: FrankaDeployConfig, model) -> str:
    """Auto-detect unnorm_key from checkpoint/dataset_statistics.json."""
    selected_key, _ = _resolve_unnorm_key(cfg, model)
    return selected_key


def initialize_model(cfg: FrankaDeployConfig):
    model = get_model(cfg)
    model.set_version(cfg.save_version)
    
    cfg.unnorm_key = _auto_detect_unnorm_key(cfg, model)

    processor = get_processor(cfg)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=cfg.proprio_dim)

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
            proprio_dim=cfg.proprio_dim,
            llm_dim=model.llm_dim,
        )
        checkpoint_path = find_checkpoint_file(str(cfg.pretrained_checkpoint), "action_id_discriminator")
        action_id_discriminator.load_state_dict(load_component_state_dict(checkpoint_path))
        action_id_discriminator = action_id_discriminator.to(torch.bfloat16).to(cfg.device)
        action_id_discriminator.eval()

    resize_size = get_image_resize_size(cfg)

    _maybe_to_device(cfg, model)

    return model, processor, action_head, proprio_projector, action_id_discriminator, resize_size


def _decode_b64_image(b64_str: str) -> np.ndarray:
    """Decode base64 image bytes into HxWx3 uint8 RGB numpy array."""
    if not isinstance(b64_str, str) or len(b64_str) == 0:
        raise ValueError("image base64 must be a non-empty string")

    try:
        img_bytes = base64.b64decode(b64_str)
    except Exception as e:
        raise ValueError(f"invalid base64: {e}")

    # Use PIL via OpenVLA utilities dependencies (PIL is already used in this repo).
    from PIL import Image
    import io

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensors or nested tensor lists to a CPU numpy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    if isinstance(value, (list, tuple)):
        return np.asarray([_to_numpy(item) for item in value], dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


class FrankaJsonServer:
    def __init__(self, cfg: FrankaDeployConfig):
        self.cfg = cfg
        (
            self.model,
            self.processor,
            self.action_head,
            self.proprio_projector,
            self.action_id_discriminator,
            self.resize_size,
        ) = initialize_model(cfg)

        set_seed_everywhere(self.cfg.seed)

        self.app = FastAPI()

        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start = time.time()
            response = await call_next(request)
            ms = (time.time() - start) * 1000
            logging.info(f'"{request.method} {request.url.path}" {response.status_code} - {ms:.2f}ms')
            return response

        self.app.post("/act")(self.act)
        self.app.get("/health")(self.health)

    async def health(self) -> Dict[str, Any]:
        return {"ok": True, "model_family": self.cfg.model_family}

    async def act(self, request: Request) -> Dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        try:
            full_b64 = body["full_image"]
            wrist_b64 = body["wrist_image"]
            goal = body.get("goal", "")
            state = body.get("state", None)

            full_rgb = _decode_b64_image(full_b64)
            wrist_rgb = _decode_b64_image(wrist_b64)

            if state is None:
                raise ValueError("state is required")
            state = np.asarray(state, dtype=np.float32)
            if state.shape != (self.cfg.proprio_dim,):
                raise ValueError(f"state must be shape ({self.cfg.proprio_dim},), got {state.shape}")

            obs = {
                "full_image": full_rgb,
                "wrist_image": wrist_rgb,
                "state": state,
            }

            action_id_pred = None
            routing_info = None
            if self.cfg.use_moe_action_head or self.cfg.use_action_id_discriminator:
                actions, action_id_pred = get_action_with_action_id(
                    cfg=self.cfg,
                    model=self.model,
                    obs=obs,
                    task_label=goal,
                    processor=self.processor,
                    action_head=self.action_head,
                    proprio_projector=self.proprio_projector,
                    noisy_action_projector=None,
                    action_id_discriminator=self.action_id_discriminator,
                    use_moe_action_head=self.cfg.use_moe_action_head,
                    use_film=self.cfg.use_film,
                    use_minivlm=self.cfg.use_minivlm,
                )
                if self.action_head is not None and hasattr(self.action_head, "get_last_routing_info"):
                    routing_info = self.action_head.get_last_routing_info()
            else:
                actions = get_action(
                    cfg=self.cfg,
                    model=self.model,
                    obs=obs,
                    task_label=goal,
                    processor=self.processor,
                    action_head=self.action_head,
                    proprio_projector=self.proprio_projector,
                    noisy_action_projector=None,
                    use_film=self.cfg.use_film,
                    use_minivlm=self.cfg.use_minivlm,
                )

            actions_np = _to_numpy(actions).astype(np.float32)
            if actions_np.ndim == 1:
                action_chunk = actions_np[None, :]
            elif actions_np.ndim == 2:
                action_chunk = actions_np
            else:
                raise ValueError(f"model returned action shape {actions_np.shape}, expected (7,) or (T, 7)")

            if action_chunk.shape[1] != 7:
                raise ValueError(f"model returned action shape {action_chunk.shape}, expected (*, 7)")

            max_chunk_len = max(1, int(self.cfg.num_open_loop_steps))
            action_chunk = action_chunk[:max_chunk_len]
            response = {
                "action": action_chunk.tolist(),
                "action_chunk_len": int(action_chunk.shape[0]),
            }
            if action_id_pred is not None:
                response["action_id_pred"] = int(_to_numpy(action_id_pred).reshape(-1)[0])
            if routing_info is not None:
                selected_expert_idx = routing_info.get("selected_expert_idx")
                stage_probs = routing_info.get("stage_probs")
                if selected_expert_idx is not None:
                    response["selected_expert_idx"] = int(_to_numpy(selected_expert_idx).reshape(-1)[0])
                if stage_probs is not None:
                    response["stage_probs"] = _to_numpy(stage_probs).reshape(-1).tolist()

            if "selected_expert_idx" in response or "action_id_pred" in response:
                logging.info(
                    "MoE route: action_id_pred=%s selected_expert_idx=%s stage_probs=%s",
                    response.get("action_id_pred"),
                    response.get("selected_expert_idx"),
                    response.get("stage_probs"),
                )
            return response

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def run(self):
        uvicorn.run(self.app, host=self.cfg.host, port=self.cfg.port, access_log=False, timeout_keep_alive=120)


@draccus.wrap()
def main(cfg: FrankaDeployConfig) -> None:
    """Entry point for Franka JSON server (use FrankaDeployConfig, not deploy.py's DeployConfig)."""
    server = FrankaJsonServer(cfg)
    server.run()


if __name__ == "__main__":
    main()
