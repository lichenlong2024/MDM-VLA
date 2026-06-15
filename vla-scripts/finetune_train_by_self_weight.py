"""
finetune.py

Fine-tunes Qwen2.5-0.5B via LoRA.
"""

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type
import torch.nn.functional as F
import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb
from prismatic.models.backbones.vision.dinosiglip_vit import DinoSigLIPViTBackbone
from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead, StageMoEActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    NUM_TOKENS
)
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
# 娣诲姞Action ID鍒ゅ埆鍣ㄧ殑瀵煎叆
from prismatic.models.action_id_discriminator import ActionIDDiscriminator, ActionIDLoss
from prismatic.models import load, load_vla
# Sane Defaults
from experiments.robot.openvla_utils import (
    load_component_state_dict
)
import safetensors
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class FinetuneConfig:
    # fmt: off
    config_file_path: str = "openvla/openvla-7b"     # Path to necessary config files of LA-Adapter
    vlm_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)
    use_minivlm: bool = False                        # 
    resum_vla_path: str = "openvla/openvla-7b"       # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("<PATH_TO_DATA_ROOT>")      # Directory containing RLDS datasets
    dataset_name: str = "libero_10_rlds_action_id"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for training 
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input
    phase1_path: str = "None"


    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0.1                       # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100000             # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200000                          # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                                 # If True, resumes from checkpoint
    resume_step: Optional[int] = 100000                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # LoRA
    use_lora: bool = False                           # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = False         # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!

    # Full Finetune
    use_fz: bool = False                             # If True, uses LoRA fine-tuning

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # revision version
    use_pro_version: bool = True                             # the version number
    phase: str = "Training"
    
    # 鏂板閰嶇疆椤癸細鏄惁鍐荤粨鍔ㄤ綔澶村弬鏁?    freeze_action_head: bool = False                 # If True, freezes action head parameters during training

    # Action ID Discriminator
    use_action_id_discriminator: bool = True         # If True, uses Action ID discriminator (鐜板湪鏄繀闇€鐨?
    train_action_id_discriminator: bool = True       # If True, trains the Action ID discriminator (鐜板湪鏄繀闇€鐨?
    action_id_discriminator_path: Optional[str] = None  # Path to pretrained Action ID discriminator
    num_action_ids: int = 18                         # Number of action IDs in the dataset


    # 鏂板閰嶇疆椤癸細浣跨敤MoE鍔ㄤ綔澶达紙鐜板湪鏄繀闇€鐨勶級
    use_moe_action_head: bool = True                 # If True, uses MoE action head instead of L1 regression head (鐜板湪鏄繀闇€鐨?
    stage_definitions: str = "0:0-2,1:3-10,2:11-17"  # Stage definitions in format "stage_id:start-end,stage_id:start-end,..."
    top_k: int = 2
    #use 3 moe head
    action_id_discriminator_path: Optional[str] = None
    action_expert_weight_path: Optional[str] = None
    proprio_weight_path: Optional[str] = None


def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def parse_stage_definitions(stage_definitions_str):
    """瑙ｆ瀽闃舵瀹氫箟瀛楃涓?""
    stage_definitions = {}
    for stage_def in stage_definitions_str.split(','):
        stage_id, action_range = stage_def.split(':')
        start, end = map(int, action_range.split('-'))
        stage_definitions[int(stage_id)] = list(range(start, end+1))
    return stage_definitions


def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.config_file_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.config_file_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_fz:
            run_id += f"+frozen+dropout-{cfg.lora_dropout}"
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id



def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def load_training_state(path: str, step: int, device: str = "cpu") -> Optional[dict]:
    """
    鍔犺浇璁粌鐘舵€侊紙optimizer, scheduler, rng 绛夛級

    Args:
        path (str): checkpoint 鐩綍璺緞
        step (int): 姝ユ暟
        device (str): 璁惧

    Returns:
        dict: 璁粌鐘舵€佸瓧鍏革紝濡傛灉鏂囦欢涓嶅瓨鍦ㄥ垯杩斿洖 None
    """
    checkpoint_path = os.path.join(path, f"training_state--{step}_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading training state: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        return state
    else:
        print(f"Warning: Training state not found at {checkpoint_path}")
        return None

def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)



def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    print(f"# trainable params in {name}: {num_params}")



def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.resum_vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)
        print('loaded!!!!!!!!!'+module_name)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)



def run_forward_pass(
    vla,
    action_head,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_proprio,
    use_film,
    num_patches,
    use_action_id_discriminator,
    use_moe_action_head,
    compute_diffusion_l1=False,
    use_pro_version=True,
    cfg=None,
    # Action ID鍒ゅ埆鍣ㄧ幇鍦ㄦ槸蹇呴渶鐨?    action_id_discriminator=None,
    action_id_loss_fn=None,
    current_step: int = 0,  # 褰撳墠璁粌姝ユ暟
    total_steps: int = 100000,  # 璁粌鎬绘鏁?    action_id_weight_init: float = 0.2,  # 鍒濆鏉冮噸
    #decay_alpha: float = 3.0,  # 鎸囨暟琛板噺绯绘暟锛堣秺澶ц“鍑忚秺蹇級
    decay_alpha: float = 15,  # 鎸囨暟琛板噺绯绘暟锛堣秺澶ц“鍑忚秺蹇級
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps (int): Number of diffusion steps (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    # VLA forward pass
    # import pdb;
    # pdb.set_trace()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=use_film,
            )

    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:,1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # Compute metrics for discrete action representation (next-token prediction)
    if not (use_l1_regression):
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)

        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=current_action_mask
            )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, 
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=current_action_mask
            )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=next_actions_mask
            )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, 
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=next_actions_mask
            )
        
        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )
        
    # Compute metrics for continuous action representations (L1 regression)
    else:
        # Get last layer hidden states
        multi_layer_hidden_states = []
        
        for item in output.hidden_states[0:]:
            # last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            # Get hidden states for text portion of prompt+response (after the vision patches)
            text_hidden_states = item[:, num_patches:-1]
            # Get hidden states for action portion of response
            batch_size = batch["input_ids"].shape[0]
            # actions_hidden_states = text_hidden_states[:, -1, :].reshape(batch_size, 1, -1).to(torch.bfloat16)
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(batch_size, 1,NUM_TOKENS, -1).to(torch.bfloat16)
            task_latten_states = item[:, :num_patches].reshape(batch_size, 1, num_patches , -1)
            all_hidden_states = torch.cat((task_latten_states, actions_hidden_states),2)
            multi_layer_hidden_states.append(all_hidden_states)
        multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim = 1)
        #import pdb;pdb.set_trace()
        # 鐜板湪濮嬬粓浣跨敤Action ID鍒ゅ埆鍣ㄨ幏鍙栭樁娈垫鐜囧苟浼犻€掔粰MoE鍔ㄤ綔澶?        action_ids = batch["action_id"].to(torch.bfloat16).to(device_id)
        proprio = batch["proprio"].to(device_id) if use_proprio else torch.zeros(batch["input_ids"].shape[0], 8).to(torch.bfloat16).to(device_id)
        

        # 鑾峰彇璇█鎸囦护淇℃伅
        input_ids = batch.get("input_ids", None)
        if input_ids is not None:
            input_ids = input_ids.to(device_id)

        # 浠嶸LA杈撳嚭涓彁鍙杢ask鍜宎ction鐗瑰緛
        # Task鐗瑰緛鏄瑙塼oken鐨勯殣钘忕姸鎬?        task_features = output.hidden_states[-1][:, :num_patches, :].to(torch.bfloat16)  # [B, num_patches, D]
        
        # Action鐗瑰緛鏄姩浣渢oken鐨勯殣钘忕姸鎬?        # 鎴戜滑闇€瑕佷粠鏂囨湰闅愯棌鐘舵€佷腑鎻愬彇鍔ㄤ綔鐩稿叧鐨勯儴鍒?        text_hidden_states = output.hidden_states[-1][:, num_patches:-1, :]  # [B, text_len, D]
        action_features = text_hidden_states[current_action_mask | next_actions_mask].reshape(
            batch_size, -1, text_hidden_states.shape[-1]).to(torch.bfloat16)  # [B, num_action_tokens, D]

        if use_action_id_discriminator and use_moe_action_head:
            # 鍒嗘敮1锛氬悓鏃朵娇鐢ㄥ垽鍒櫒鍜孧oE 鈫?鏅鸿兘璺敱
            # 4.1 杩愯鍒ゅ埆鍣ㄨ幏鍙栧姩浣滄鐜囷紙鐢ㄤ簬MoE璺敱锛?            action_probs, confidence, logits = action_id_discriminator(
                pixel_values=batch["pixel_values"].to(device_id, dtype=torch.bfloat16),
                proprio=proprio,
                input_ids=batch["input_ids"].to(device_id) if input_ids is not None else None,
                task_features=task_features,
                action_features=action_features
            )
            # 4.2 MoE鍔ㄤ綔澶撮娴嬶紙浼犻€掑垽鍒櫒杈撳嚭鐨勮矾鐢辨鐜囷級
            predicted_actions = action_head.module.predict_action(
                actions_hidden_states=multi_layer_hidden_states,
                proprio=proprio if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                phase=cfg.phase,
                action_probs=action_probs, # 鍏抽敭锛氫紶閫掑垽鍒櫒鐨勫姩浣滄鐜囩敤浜庤矾鐢?
                use_moe=use_moe_action_head
            )
            # 4.3 璁＄畻鍒ゅ埆鍣ㄦ崯澶憋紙浠呰鍒嗘敮闇€瑕侊級
            action_id_loss = action_id_loss_fn(logits, batch["action_id"].to(device_id, dtype=torch.bfloat16),
                                               confidence)

        elif use_moe_action_head:
            # 鍒嗘敮2锛氫粎浣跨敤MoE 鈫?鍧囧寑鍒嗗竷璺敱锛堝師else鍒嗘敮鐨勪竴閮ㄥ垎锛?            predicted_actions = action_head.module.predict_action(
                actions_hidden_states=multi_layer_hidden_states,
                proprio=proprio if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                phase=cfg.phase,
                action_probs=None,
                use_moe = use_moe_action_head
                # 鍏抽敭锛氫笉浼犻€抯tage_probs锛孧oE鍐呴儴浼氫娇鐢ㄥ潎鍖€鍒嗗竷璺敱
            )

        else:
            # 鍒嗘敮3锛氫袱鑰呴兘涓嶇敤 鈫?甯歌L1鍥炲綊鍔ㄤ綔澶达紙鍘焑lse鍒嗘敮鐨勫彟涓€閮ㄥ垎锛?            predicted_actions = action_head.module.predict_action(
                actions_hidden_states=multi_layer_hidden_states,
                proprio=proprio if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                phase=cfg.phase
            )

        # -------------------------- 5. 璁＄畻鏍稿績L1鎹熷け --------------------------
        ground_truth_actions = batch["actions"].to(device_id, dtype=torch.bfloat16)
        core_loss = torch.nn.L1Loss()(predicted_actions, ground_truth_actions)

        total_loss = core_loss

        training_progress = current_step / total_steps
        if current_step % 100 == 0:  # 姣?00姝ユ墦鍗颁竴娆?            print(f"Training Progress: {training_progress:.2%} ({current_step}/{total_steps})")


        # ========== MoE aux loss锛堥潪甯稿急 & 鍓嶆湡锛?==========
        if use_moe_action_head and cfg.phase == "Training":

            if training_progress < 0.002:
                moe_aux_weight = 0.02
            else:
                moe_aux_weight = 0.0  # loss 閫€鍦猴紝浣?expert 浠嶅弬涓庝富 loss

            moe_aux_loss = moe_aux_weight * action_head.module.get_aux_loss()
            total_loss = total_loss + moe_aux_loss

        # ========== Action ID discriminator锛?0% 閫€鍦猴級 ==========
        # ========== Action ID discriminator ==========
        if cfg.train_action_id_discriminator and action_id_loss is not None:

            if training_progress < 0.1:
                action_id_weight = action_id_weight_init * torch.exp(
                    torch.tensor(-decay_alpha * training_progress, device=device_id)
                )
            else:
                # 鍐荤粨鍙傛暟锛堝彧鎵ц涓€娆★級
                if not getattr(action_id_discriminator, "_frozen", False):
                    for p in action_id_discriminator.parameters():
                        p.requires_grad_(False)
                    action_id_discriminator.eval()  # 鍙€夛細鍐荤粨 BN / Dropout
                    action_id_discriminator._frozen = True

                action_id_weight = 0.0

            weighted_action_id_loss = action_id_weight * action_id_loss
            total_loss = total_loss + weighted_action_id_loss

        # -------------------------- 7. 鏀堕泦鎸囨爣锛堝垎鎯呭喌鏇存柊锛?--------------------------
        metrics = {
            "loss_value": total_loss.item(),  # 鎬绘崯澶憋紙鐢ㄤ簬鏃ュ織锛?            "core_l1_loss": core_loss.item()  # 鏍稿績鍔ㄤ綔L1鎹熷け锛堜究浜庡崟鐙洃鎺э級
        }

        # 鉁?娣诲姞 core_loss 鐢ㄤ簬 W&B 骞虫粦鐩戞帶
        metrics["core_loss"] = core_loss.item()

        if use_moe_action_head:
            routing_info = action_head.module.get_last_routing_info()
            selected_expert_idx = routing_info.get("selected_expert_idx")
            if selected_expert_idx is not None:
                per_sample_l1 = torch.abs(predicted_actions - ground_truth_actions).mean(dim=(1, 2))
                num_experts = len(action_head.module.stage_experts)
                batch_size = int(selected_expert_idx.shape[0])
                for expert_idx in range(num_experts):
                    expert_mask = (selected_expert_idx == expert_idx)
                    expert_count = int(expert_mask.sum().item())
                    metrics[f"expert_{expert_idx}_count"] = expert_count
                    metrics[f"expert_{expert_idx}_usage_ratio"] = (
                        float(expert_count) / float(batch_size) if batch_size > 0 else 0.0
                    )
                    if expert_count > 0:
                        metrics[f"expert_{expert_idx}_conditional_l1_loss"] = (
                            per_sample_l1[expert_mask].mean().item()
                        )

        # 7.0 MoE杈呭姪鎹熷け鍜岃礋杞藉潎琛℃寚鏍?        if use_moe_action_head and cfg.phase == "Training":
            metrics["moe_aux_loss"] = moe_aux_loss.item()  # 鉁?宸叉湁

            # 鑾峰彇璐熻浇鍧囪　鎸囨爣
            load_balance_metrics = action_head.module.get_load_balance_metrics()
            metrics.update({
                "moe_importance_loss": load_balance_metrics.get('importance_loss', 0.0),
                "moe_load_loss": load_balance_metrics.get('load_loss', 0.0),
            })

            # 涓撳浣跨敤鎯呭喌锛堝彲閫夛細鐢ㄤ簬璇︾粏鐩戞帶锛?            expert_usage = load_balance_metrics.get('expert_usage', [])
            if expert_usage:
                for i, usage in enumerate(expert_usage):
                    metrics[f"expert_{i}_usage"] = usage
        else:
            # 濡傛灉涓嶄娇鐢?MoE 鎴栦笉鍦ㄨ缁冮樁娈碉紝璁剧疆涓?0
            metrics["moe_aux_loss"] = 0.0

        # 鉁?娣诲姞 weighted_action_id_loss
        if cfg.train_action_id_discriminator and training_progress < 0.1:
            metrics["weighted_action_id_loss"] = weighted_action_id_loss.item()
        else:
            metrics["weighted_action_id_loss"] = 0.0

        # 7.1 璇︾粏L1鎹熷け锛堝綋鍓嶅姩浣?+ 涓嬩竴涓姩浣滐級
        if use_l1_regression:
            # 鎷嗗垎褰撳墠鍔ㄤ綔鍜屼笅涓€涓姩浣?            ground_truth_curr = ground_truth_actions[:, 0]
            predicted_curr = predicted_actions[:, 0]
            ground_truth_next = ground_truth_actions[:, 1:]
            predicted_next = predicted_actions[:, 1:]

            # 璁＄畻璇︾粏L1鎹熷け
            curr_l1 = torch.nn.L1Loss()(ground_truth_curr, predicted_curr)
            next_l1 = torch.nn.L1Loss()(ground_truth_next,
                                        predicted_next) if ground_truth_next.numel() > 0 else torch.tensor(0.0,
                                                                                                           device=device_id)

            # 鏇存柊鎸囨爣
            metrics.update({
                "curr_action_l1_loss": curr_l1.item(),
                "next_actions_l1_loss": next_l1.item()
            })

            # 鍘熼€昏緫锛氱壒瀹氭潯浠朵笅鎵撳嵃璇︾粏鎹熷け
            if compute_diffusion_l1:
                print(f"Current action L1 loss: {curr_l1.item():.4f}")
                if ground_truth_next.numel() > 0:
                    print(f"Next actions L1 loss: {next_l1.item():.4f}")

        # 7.2 鍒ゅ埆鍣ㄧ浉鍏虫寚鏍囷紙浠呭綋浣跨敤鍒ゅ埆鍣ㄦ椂璁板綍锛?        if use_action_id_discriminator:
            action_id_acc = (logits.argmax(dim=-1) == batch["action_id"].to(device_id)).float().mean()
            avg_conf = confidence.mean()
            metrics.update({
                "action_id_loss": action_id_loss.item() if action_id_loss is not None else 0.0,
                "action_id_accuracy": action_id_acc.item(),
                "avg_confidence": avg_conf.item()
            })

        # -------------------------- 8. 杩斿洖缁撴灉 --------------------------
        return total_loss, metrics



def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics



def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)



def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    distributed_state,
    new_state_dict,
    # 娣诲姞Action ID鍒ゅ埆鍣ㄥ弬鏁?    action_id_discriminator=None,
    optimizer=None,  # <-- 鏂板
    scheduler=None,
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        train_dataset (RLDSDataset): Training dataset.
        distributed_state (PartialState): Distributed training state.
        action_id_discriminator (nn.Module): Action ID discriminator module.

    Returns:
        None.
    """
    # Determine checkpoint paths and naming
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)

        if cfg.use_fz:
            vla.module.save_pretrained(checkpoint_dir) # directly save checkpoint without lora
        else:
            vla.module.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(),
                checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )

        if cfg.use_l1_regression and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        if cfg.use_film:
            # To be safe, just save the entire vision backbone (not just FiLM components)
            torch.save(
                vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )
            
        # 淇濆瓨Action ID鍒ゅ埆鍣?        if cfg.use_action_id_discriminator and action_id_discriminator is not None:
            torch.save(
                action_id_discriminator.module.state_dict(), checkpoint_dir / f"action_id_discriminator--{checkpoint_name_suffix}"
            )
            # ============== 鏂板锛氫繚瀛樿缁冪姸鎬?==============
        training_state = {
            'step': log_step,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
        }
        if optimizer is not None:
            training_state['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            training_state['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(training_state, checkpoint_dir / f"training_state--{checkpoint_name_suffix}")
        print(f"Saved training state (optimizer, scheduler, rng) for Step {log_step}")
    # Wait for model components to be saved
    dist.barrier()

    # Merge LoRA weights into base model and save resulting model checkpoint
    # Note: Can be very slow on some devices; if so, we recommend merging offline
    if cfg.use_lora and cfg.merge_lora_during_training:
        if cfg.use_minivlm:
            config = AutoConfig.from_pretrained("<PATH_TO_VLA_CONFIG>/config.json")
            base_vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)  # Create a new model with configuration, the parameters are randomly initialized
            # print(new_state_dict['action_queries.weight'])
            new_state_dict['action_queries.weight'] = vla.state_dict()['module.base_model.model.action_queries.weight'].cpu()
            missing_keys, unexpected_keys = base_vla.load_state_dict(new_state_dict, strict=False)
            
        else:
            base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, trust_remote_code=False
        )

        # import pdb ;pdb.set_trace()

        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")
        
        # Wait for merged model to be saved
        dist.barrier()
    # 淇濆瓨璁粌鐘舵€?

def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    start_step,
    distributed_state,
    val_time_limit,
    # 娣诲姞Action ID鍒ゅ埆鍣ㄥ弬鏁?    action_id_discriminator=None,
) -> None:

    val_start_time = time.time()
    vla.eval()
    if action_id_discriminator is not None:
        action_id_discriminator.eval()
    val_batches_count = 0

    # List to store validation metrics
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                use_moe_action_head=cfg.use_moe_action_head,
                compute_diffusion_l1=True,
                use_pro_version=cfg.use_pro_version,
                cfg=cfg,
                # 娣诲姞Action ID鍒ゅ埆鍣ㄥ弬鏁?                action_id_discriminator=action_id_discriminator,
                action_id_loss_fn=ActionIDLoss() if action_id_discriminator is not None else None,
                current_step=start_step,  # 鍏ㄥ眬鏃ュ織姝ユ暟浣滀负鍔ㄦ€佹潈閲嶇殑褰撳墠姝ユ暟
                total_steps=cfg.max_steps,  # 鏈€澶ф搴︽鏁颁綔涓烘€绘鏁?                action_id_weight_init=0.2,
                decay_alpha=3.0,
            )

            # Add the loss value to the metrics
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut testing on validation set short if it exceeds time limit
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:

    global RAW_STATE_DICT

    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.config_file_path = cfg.config_file_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.config_file_path}` on `{cfg.dataset_name}`")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Initialize wandb logging
    if distributed_state.is_main_process:
        wandb.init(project=cfg.wandb_project, name=f"ft+{run_id}", mode="online")

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect

    if model_is_on_hf_hub(cfg.config_file_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.config_file_path)
        # Overwrite VLA path
        cfg.config_file_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


    # Update config.json and sync model files
    if distributed_state.is_main_process:
        update_auto_map(cfg.config_file_path)
        check_model_logic_mismatch(cfg.config_file_path)

    # Wait for model files to be synced
    dist.barrier()

    # Load processor and VLA
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    processor = AutoProcessor.from_pretrained(cfg.config_file_path, trust_remote_code=True)

    if cfg.use_minivlm:
        hf_token = ''
        if 'prism-qwen25-extra-dinosiglip-224px-0_5b' in cfg.vlm_path:
            
            vlm = load(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
        else:
            vlm = load_vla(
                cfg.vlm_path,
                hf_token=hf_token,
                load_for_training=True,
                )
        config = AutoConfig.from_pretrained("<PATH_TO_VLA_CONFIG>/config.json")
        print("line788 鍔犺浇config")
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16).to(device_id)  # Create a new model with configuration, the parameters are randomly initialized
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.shape}")
        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
            ]

        def rename_state_dict_keys(state_dict, replace_map):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                for old, new in replace_map:
                    if old in new_k:
                        new_k = new_k.replace(old, new)
                new_state_dict[new_k] = v
            return new_state_dict
        
        old_state_dict = vlm.state_dict()
        RAW_STATE_DICT = rename_state_dict_keys(old_state_dict, replace_map)
    
        missing_keys, unexpected_keys = vla.load_state_dict(RAW_STATE_DICT, strict=False)
        del old_state_dict

    else:
        RAW_STATE_DICT ={}
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=False,
            ).to(device_id)

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # vla.set_version(cfg.version)

    if cfg.use_lora:
        if cfg.resume:

            adapter_dir = Path(cfg.resum_vla_path) / "lora_adapter"

            # 姝ラ1: 鐢?"all-linear" 鍒涘缓缁撴瀯
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=2 * cfg.lora_rank,
                lora_dropout=cfg.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)

            # 姝ラ2: 鍔犺浇鏉冮噸
            from safetensors.torch import load_file
            weights_path = adapter_dir / "adapter_model.safetensors"
            state_dict = load_file(str(weights_path))
            vla.load_state_dict(state_dict, strict=False)

            print("鉁?Loaded LoRA weights from checkpoint")
        else:
            # 闈瀝esume鏃讹細姝ｅ父鍒涘缓PeftModel
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=2 * cfg.lora_rank,
                lora_dropout=cfg.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)

        # 纭繚action_queries鍙缁?        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True
        vla.print_trainable_parameters()
    else:
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        # Wrap vision backbone with FiLM wrapper
        # Important: For this, must specify `vla.model.vision_backbone` instead of just `vla.vision_backbone`, since the
        # latter would cause the new wrapped backbone to be saved as a new attribute of `vla` instead of overwriting the
        # original one (due to the LoRA wrapper)
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.config_file_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # If applicable, instantiate proprio projector
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
            to_bf16=True,
        )
        if cfg.proprio_weight_path:
            proprio_weight = torch.load(cfg.proprio_weight_path, map_location=f"cuda:{device_id}")
            proprio_weight_state_dict =remove_ddp_in_checkpoint(proprio_weight)
            proprio_projector.module.load_state_dict(proprio_weight_state_dict)
            print(f"Loaded proprio projector from {cfg.proprio_weight_path}")
    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:
        if cfg.use_moe_action_head:
            # 鐜板湪濮嬬粓浣跨敤MoE鍔ㄤ綔澶?            stage_definitions = parse_stage_definitions(cfg.stage_definitions)
            action_head = init_module(
                StageMoEActionHead,
                "action_head",
                cfg,
                device_id,
                {
                    "input_dim": vla.module.llm_dim,
                    "hidden_dim": vla.module.llm_dim,
                    "action_dim": ACTION_DIM,
                    "use_pro_version": cfg.use_pro_version,
                    "stage_definitions": stage_definitions,  # 浼犻€掗樁娈靛畾涔?                    "num_action_ids":cfg.num_action_ids,
                },
                to_bf16=True,
            )
            # if cfg.action_expert_weight_path:
            #     action_head_dict = torch.load(cfg.action_expert_weight_path, map_location=f"cuda:{device_id}")
            #     action_new_head_dict =remove_ddp_in_checkpoint(action_head_dict)
            #     action_head.module.load_state_dict(action_new_head_dict)
            #     print(f"Loaded action head from {cfg.action_expert_weight_path}")
            if cfg.action_expert_weight_path:
                # 1. 鍔犺浇鍗曚釜 expert 鐨勯璁粌鏉冮噸锛堝厛鍔犺浇鍒?CPU锛?                expert_state_dict = torch.load(cfg.action_expert_weight_path, map_location="cpu")

                # 2. 濡傛灉淇濆瓨鏃剁敤浜?DDP锛堝甫 module. 鍓嶇紑锛夛紝鍘绘帀鍓嶇紑
                expert_state_dict = remove_ddp_in_checkpoint(expert_state_dict)
                print("鉁?宸插幓鎺夐璁粌鏉冮噸閿悕鐨勩€宮odule.銆嶅墠缂€")

                # 3. 璋冭瘯锛氭墦鍗伴敭鍚嶆牱渚?                print("澶勭悊鍚庨璁粌鏉冮噸閿悕鏍蜂緥(鍓?涓?:", list(expert_state_dict.keys())[:3])
                sample_expert = action_head.module.stage_experts[0]
                expert_keys = sample_expert.state_dict().keys()
                print("MoE涓撳閿悕鏍蜂緥(鍓?涓?:", list(expert_keys)[:3])

                # 4. 楠岃瘉閿悕鏄惁瀹屽叏鍖归厤
                expert_key_set = set(expert_keys)
                loaded_key_set = set(expert_state_dict.keys())

                if expert_key_set != loaded_key_set:
                    missing = expert_key_set - loaded_key_set
                    unexpected = loaded_key_set - expert_key_set
                    print(f"鉂?閿悕涓嶅尮閰?")
                    print(f"  缂哄け鐨勯敭锛堜笓瀹堕渶瑕佷絾鏉冮噸娌℃湁锛? {list(missing)[:5]}...")
                    print(f"  澶氫綑鐨勯敭锛堟潈閲嶆湁浣嗕笓瀹朵笉闇€瑕侊級: {list(unexpected)[:5]}...")
                    raise ValueError("棰勮缁?expert 鏉冮噸涓?MoE 涓撳缁撴瀯涓嶄竴鑷达紝璇锋鏌ワ紒")

                # 5. 灏嗚 expert 鏉冮噸閫愪釜鍔犺浇鍒版墍鏈?stage_experts 涓?                for idx, expert in enumerate(action_head.module.stage_experts):
                    expert.load_state_dict(expert_state_dict, strict=True)

                print(f"鉁?鎴愬姛灏?{cfg.action_expert_weight_path} 鐨勬潈閲嶅垵濮嬪寲鍒?"
                      f"{len(action_head.module.stage_experts)} 涓?MoE 涓撳涓?)
        else:
            # 濡傛灉浣跨敤L1鍥炲綊浣嗕笉浣跨敤MoE锛屽垱寤烘櫘閫氱殑L1鍥炲綊鍔ㄤ綔澶?            action_head = init_module(
                L1RegressionActionHead,
                "action_head",
                cfg,
                device_id,
                {
                    "input_dim": vla.module.llm_dim,
                    "hidden_dim": vla.module.llm_dim,
                    "action_dim": ACTION_DIM,
                    "use_pro_version": cfg.use_pro_version,
                },
                to_bf16=True,
            )
            if cfg.action_expert_weight_path:  # 鍗曚笓瀹舵潈閲嶉厤缃」锛堝彲澶嶇敤鍘焎fg瀛楁锛屽缓璁涔夊寲鍛藉悕锛?                # 1. 鍔犺浇鍗曚笓瀹堕璁粌鏉冮噸锛堝厛鍔犺浇鍒癈PU锛岄伩鍏嶆樉瀛樺崰鐢級
                single_expert_state_dict = torch.load(
                    cfg.action_expert_weight_path,
                    map_location="cpu"
                )
                print(f"馃摜 宸插姞杞?L1 鍥炲綊鍗曚笓瀹舵潈閲嶆枃浠? {cfg.action_expert_weight_path}")

                # 2. 澶勭悊DDP璁粌鐨勬潈閲嶏紙鍘绘帀module.鍓嶇紑锛?                #single_expert_state_dict = remove_ddp_in_checkpoint(single_expert_state_dict)
                print("鉁?宸插幓鎺夊崟涓撳鏉冮噸閿悕鐨勩€宮odule.銆嶅墠缂€")

                # 3. 璋冭瘯锛氭墦鍗伴敭鍚嶆牱渚嬶紝渚夸簬鎺掓煡鍖归厤闂
                print("澶勭悊鍚庡崟涓撳鏉冮噸閿悕鏍蜂緥(鍓?涓?:", list(single_expert_state_dict.keys())[:3])
                single_expert_keys = action_head.state_dict().keys()  # 鐩存帴鍙栧崟涓撳鐨勯敭鍚?                print("L1鍥炲綊鍗曚笓瀹堕敭鍚嶆牱渚?鍓?涓?:", list(single_expert_keys)[:3])

                # 4. 涓ユ牸楠岃瘉閿悕鍖归厤鎬э紙閬垮厤缁撴瀯涓嶄竴鑷村鑷村姞杞藉け璐ワ級
                single_expert_key_set = set(single_expert_keys)
                loaded_key_set = set(single_expert_state_dict.keys())

                if single_expert_key_set != loaded_key_set:
                    missing_keys = single_expert_key_set - loaded_key_set
                    unexpected_keys = loaded_key_set - single_expert_key_set
                    print(f"鉂?L1鍥炲綊鍗曚笓瀹舵潈閲嶉敭鍚嶄笉鍖归厤!")
                    print(f"  缂哄け鐨勯敭锛堝崟涓撳闇€瑕佷絾鏉冮噸娌℃湁锛? {list(missing_keys)[:5]}...")
                    print(f"  澶氫綑鐨勯敭锛堟潈閲嶆湁浣嗗崟涓撳涓嶉渶瑕侊級: {list(unexpected_keys)[:5]}...")
                    # 鍙€夛細闈炰弗鏍兼ā寮忥紙濡傞渶鍏煎閮ㄥ垎鏉冮噸锛屾敞閲妑aise鍚敤涓嬫柟锛?                    # print("鈿狅笍 鍚敤闈炰弗鏍兼ā寮忓姞杞芥潈閲嶏紙浠呭尮閰嶅瓨鍦ㄧ殑閿級")
                    # action_head.load_state_dict(single_expert_state_dict, strict=False)
                    raise ValueError("棰勮缁冨崟涓撳鏉冮噸涓嶭1鍥炲綊鍔ㄤ綔澶寸粨鏋勪笉涓€鑷达紝璇锋鏌ワ紒")

                # 5. 鏍稿績锛氬皢鏉冮噸鍔犺浇鍒板敮涓€鐨凩1鍥炲綊涓撳涓紙鏃犲惊鐜紝鐩存帴鍔犺浇锛?                action_head.load_state_dict(single_expert_state_dict, strict=True)
                print("鉁?L1鍥炲綊鍗曚笓瀹舵潈閲嶅姞杞藉畬鎴愶紙涓ユ牸妯″紡锛?)


    # 濡傛灉鍚敤锛屽疄渚嬪寲Action ID鍒ゅ埆鍣?    action_id_discriminator = None
    action_id_loss_fn = None
    vision_backbone = DinoSigLIPViTBackbone(
        vision_backbone_id="dinosiglip-vit-so-224px",
        image_resize_strategy="resize-naive",
    )
    if cfg.use_action_id_discriminator:
        # 浣跨敤init_module鍒涘缓ActionIDDiscriminator浠ョ‘淇濇纭殑鏁版嵁绫诲瀷杞崲
        action_id_discriminator = init_module(
            ActionIDDiscriminator,
            "action_id_discriminator",
            cfg,
            device_id,
            {
                "num_action_ids": cfg.num_action_ids,
                "vision_backbone": vision_backbone,  # <--- 鍏抽敭锛氱‘淇濊繖涓€琛屽瓨鍦ㄤ笖姝ｇ‘
                "llm_dim": vla.module.llm_dim,
                "proprio_dim": PROPRIO_DIM
            },
            to_bf16=True,  # 纭繚杞崲涓篵float16
        )
            
        # 濡傛灉鎻愪緵浜嗛璁粌鐨勫垽鍒櫒璺緞锛屽垯鍔犺浇鏉冮噸
        if cfg.action_id_discriminator_path :
            # # load pt鏉冮噸
            # action_id_dict = torch.load(cfg.action_id_discriminator_path, map_location=f"cuda:{device_id}")
            # action_id_discriminator.module.load_state_dict(action_id_dict)
            #load pth 鏉冮噸
            ckpt = torch.load(cfg.action_id_discriminator_path, map_location="cpu")

            state_dict = ckpt["discriminator_state_dict"]
            action_id_discriminator.module.load_state_dict(
                state_dict,
                strict=True  # 馃憟 鍏抽敭
            )
            print(f"Loaded Action ID Discriminator from {cfg.action_id_discriminator_path}")


    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings

    # Instantiate optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]

    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
        
    # 鐜板湪濮嬬粓娣诲姞Action ID鍒ゅ埆鍣ㄥ弬鏁板埌浼樺寲鍣?    if cfg.use_action_id_discriminator and cfg.train_action_id_discriminator:
        trainable_params += [param for param in action_id_discriminator.parameters() if param.requires_grad]
        
    # 濡傛灉璁剧疆浜嗗喕缁撳姩浣滃ご锛屽垯浠庝紭鍖栧櫒鍙傛暟鍒楄〃涓Щ闄ゅ姩浣滃ご鍙傛暟
    if cfg.freeze_action_head and cfg.use_l1_regression:
        # 浠巘rainable_params涓Щ闄ction_head鐨勫弬鏁?        trainable_params = [param for param in trainable_params if param not in action_head.parameters()]
        # 鍐荤粨action_head鐨勫弬鏁?        for param in action_head.parameters():
            param.requires_grad = False
        print("Action head parameters have been frozen.")


    if cfg.use_action_id_discriminator and not cfg.train_action_id_discriminator:
        # 鍐荤粨action_id_discriminator鐨勫弬鏁?        for param in action_id_discriminator.parameters():
            param.requires_grad = False
        print("Action ID discriminator parameters have been frozen.")

    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)


    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    # 1. MultiStepLR
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
        gamma=0.1,  # Multiplicative factor of learning rate decay
    )
    # ============== 鏂板锛氬姞杞借缁冪姸鎬?==============
    if cfg.resume:
        training_state = load_training_state(cfg.resum_vla_path, cfg.resume_step, device='cpu')
        if training_state is not None:
            # if 'optimizer_state_dict' in training_state:
            #     optimizer.load_state_dict(training_state['optimizer_state_dict'])
            #     print("鉁?Loaded optimizer state from checkpoint")
            if 'scheduler_state_dict' in training_state:
                scheduler.load_state_dict(training_state['scheduler_state_dict'])
                print("鉁?Loaded scheduler state from checkpoint")
            if 'rng_state' in training_state:
                torch.set_rng_state(training_state['rng_state'])
                print("鉁?Loaded CPU RNG state from checkpoint")
            if 'cuda_rng_state' in training_state:
                torch.cuda.set_rng_state(training_state['cuda_rng_state'])
                print("鉁?Loaded CUDA RNG state from checkpoint")
        else:
            print("鈿狅笍 Warning: No training state found, optimizer and scheduler will start fresh")


    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    use_wrist_image = cfg.num_images_in_input > 1

    # Create training and optional validation datasets
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        use_minivlm=cfg.use_minivlm
        )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create collator and dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    print('Len of dataloader: ', len(dataloader))
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "core_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "moe_aux_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "weighted_action_id_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }
    # Start training
    start_step = cfg.resume_step if cfg.resume else 0

    with tqdm.tqdm(total=cfg.max_steps, initial=start_step, leave=False) as progress:
        vla.train()
        # 鐜板湪濮嬬粓灏咥ction ID鍒ゅ埆鍣ㄨ缃负璁粌妯″紡
        if cfg.use_action_id_discriminator:
            action_id_discriminator.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            # Compute training metrics and loss
            compute_diffusion_l1 = (cfg.use_l1_regression and batch_idx % cfg.diffusion_sample_freq == 0) or (cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0)
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                use_moe_action_head=cfg.use_moe_action_head,
                use_action_id_discriminator=cfg.use_action_id_discriminator,
                compute_diffusion_l1=compute_diffusion_l1,
                use_pro_version=cfg.use_pro_version,
                cfg=cfg,
                # 鐜板湪濮嬬粓浼犻€扐ction ID鍒ゅ埆鍣ㄧ浉鍏冲弬鏁?                action_id_discriminator=action_id_discriminator if cfg.use_action_id_discriminator else None,
                action_id_loss_fn=ActionIDLoss() if (action_id_discriminator is not None) else None,
                current_step=start_step,  # 鍏ㄥ眬鏃ュ織姝ユ暟浣滀负鍔ㄦ€佹潈閲嶇殑褰撳墠姝ユ暟
                total_steps=cfg.max_steps,  # 鏈€澶ф搴︽鏁颁綔涓烘€绘鏁?                action_id_weight_init=0.2,
                decay_alpha=3.0,
            )

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Store recent train metrics
            for metric_name, value in metrics.items():
                if metric_name not in recent_metrics:
                    recent_metrics[metric_name] = deque(maxlen=cfg.grad_accumulation_steps)
                recent_metrics[metric_name].append(value)

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            # Push Metrics to W&B (every wandb_log_freq gradient steps)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            start_step =log_step
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)

            # [If applicable] Linearly warm up learning rate from 10% to 100% of original
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                # Log the learning rate
                # Make sure to do this AFTER any learning rate modifications (e.g., warmup/decay)
                wandb.log(
                    {
                        "VLA Train/Learning Rate": scheduler.get_last_lr()[0],
                    },
                    step=log_step,
                )

            # Optimizer and LR scheduler step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=None,
                    action_head=action_head,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                    new_state_dict=RAW_STATE_DICT,
                    # 鐜板湪濮嬬粓淇濆瓨Action ID鍒ゅ埆鍣?                    action_id_discriminator=action_id_discriminator,
                    optimizer=optimizer,  # <-- 鏂板
                    scheduler=scheduler,  # <-- 鏂板
                )
                # import pdb;pdb.set_trace()
            # Test model on validation set
            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=start_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                    # 鐜板湪濮嬬粓浼犻€扐ction ID鍒ゅ埆鍣ㄩ獙璇佸弬鏁?                    action_id_discriminator=action_id_discriminator,
                )
                # Set model back to training mode after validation
                vla.train()
                # 鐜板湪濮嬬粓灏咥ction ID鍒ゅ埆鍣ㄨ缃负璁粌妯″紡
                action_id_discriminator.train()

            # Stop training when max_steps is reached
            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()

