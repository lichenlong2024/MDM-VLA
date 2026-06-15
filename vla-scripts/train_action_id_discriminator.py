# train_action_id_discriminator.py
"""
璁粌Action ID鍒ゅ埆鍣ㄧ殑瀹屾暣鑴氭湰
"""
from huggingface_hub import HfApi, snapshot_download
import os
import time
from collections import deque
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import numpy as np
from prismatic.models import load
from prismatic.models.action_id_discriminator import ActionIDDiscriminator, ActionIDLoss
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM
from prismatic.models.action_heads import L1RegressionActionHead, StageMoEActionHead
from prismatic.models.projectors import ProprioProjector
from prismatic.models.backbones.vision.dinosiglip_vit import DinoSigLIPViTBackbone
from transformers import AutoModelForVision2Seq, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
import wandb
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask
)
# 璁剧疆鐜鍙橀噺浠ラ伩鍏嶄氦浜掑紡鎻愮ず
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"
os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "true"
from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map
)
from prismatic.models import load, load_vla
import os
import time
import torch
import wandb
import timm
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# ==============================================================================
# 1. 閰嶇疆鍙傛暟 (Configuration)
#    鎵€鏈夊彲璋冩暣鐨勫弬鏁伴兘闆嗕腑鍦ㄨ繖閲岋紝渚夸簬绠＄悊
# ==============================================================================
CONFIG = {
    # 璺緞 (Paths)
    "config_path": "<PATH_TO_VLA_CONFIG>",
    "vlm_path": "<PATH_TO_PRETRAINED_VLM>",
    "data_root_dir": "<PATH_TO_DATA_ROOT>",
    "dataset_name": "libero_bread_action_id",
    "save_dir": "<PATH_TO_DISCRIMINATOR_OUTPUT_DIR>",
    "resume_checkpoint_path": None,
    #"resume_checkpoint_path": "<PATH_TO_DISCRIMINATOR_CHECKPOINT>",
    # 妯″瀷鍙傛暟 (Model Parameters)
    "num_images_in_input": 2,
    "ACTION_DIM": 8,  # 鎴栦綘鐨勫姩浣滅淮搴?    "proprio_dim": 8,
    "num_action_ids": 18,
    "vision_backbone_id": "dinosiglip-vit-so-224px",
    "image_resize_strategy": "resize-naive",
    "image_size": 224,
    "NUM_PATCHES": 196,  # 鍏抽敭鍙傛暟锛屽繀椤讳笌VLA妯″瀷鐨勮瑙夎緭鍑轰竴鑷?
    # 璁粌瓒呭弬鏁?(Training Hyperparameters)
    "batch_size": 12,
    "learning_rate": 1e-4,
    "max_steps": 10000,
    "grad_accumulation_steps": 1,
    "weight_decay": 1e-4,
    "lr_warmup_steps": 1000,
    "num_steps_before_decay": 8000,
    "lr_decay_gamma": 0.1,

    # 楠岃瘉涓庝繚瀛?(Validation & Saving)
    "val_freq": 5000,
    "val_time_limit": 180,
    "save_freq":20,

    # 璁粌绛栫暐 (Training Strategy)
    "use_vla_model": True,
    "use_moe_action_head": True,
    "freeze_vla_action_head": True,
    "freeze_vision_backbone": True,

    # 鏃ュ織 (Logging)
    "wandb_project": "libero_bread_action_id",
    "wandb_run_name": "train-action-id-discriminator-bread",
    "resume": False,
}


# ==============================================================================
# 2. 杈呭姪鍑芥暟 (Helper Functions)
#    灏嗛噸澶嶆€у伐浣滄ā鍧楀寲锛屼娇涓昏缁冨惊鐜洿娓呮櫚
# ==============================================================================

def setup_environment(cfg: dict) -> torch.device:
    """璁剧疆璁粌鐜锛屽寘鎷澶囬€夋嫨鍜岀洰褰曞垱寤恒€?""
    os.makedirs(cfg["save_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    return device


def setup_wandb(cfg: dict):
    """鍒濆鍖?Weights & Biases 鐢ㄤ簬瀹為獙璺熻釜銆?""
    try:
        wandb.login(anonymous="allow")
    except Exception as e:
        print(f"[!] Warning: Could not login to wandb: {e}")
    wandb.init(project=cfg["wandb_project"], name=cfg["wandb_run_name"], config=cfg)
    print("[*] Wandb initialized.")



# 鍋囪杩欎簺鏄綘宸叉湁鐨勫叾浠栧鍏?# from your_module import StageMoEActionHead, L1RegressionActionHead, ProprioProjector
# from your_module import DinoSigLIPViTBackbone, ActionIDDiscriminator

# 鍏ㄥ眬鍙橀噺锛岀敤浜庡瓨鍌ㄥ師濮嬬殑銆佽浆鎹㈠悗鐨勭姸鎬佸瓧鍏革紙濡傛灉闇€瑕佸湪澶栭儴浣跨敤锛?RAW_STATE_DICT = {}


def create_models(cfg: dict, device: torch.device) -> tuple[
    nn.Module, nn.Module, nn.Module, nn.Module, PrismaticProcessor]:
    """
    鍒涘缓骞惰繑鍥炴墍鏈夋ā鍨嬬粍浠躲€?    - vla: VLA 鍩虹妯″瀷 (涓庡弬鑰冧唬鐮侀厤缃畬鍏ㄥ榻?
    - action_head: 鐙珛鐨勫姩浣滈娴嬪ご
    - proprio_projector: 鐙珛鐨勬湰浣撴劅鍙楁姇褰卞櫒
    - discriminator: 寰呰缁冪殑鍔ㄤ綔ID鍒ゅ埆鍣?    - processor: 鏁版嵁澶勭悊鍣?    """
    print("\n[*] Creating models...")

    # --- 鏍稿績淇敼寮€濮?---

    # 1. 娉ㄥ唽 OpenVLA 妯″瀷鍜屽鐞嗗櫒鍒?Hugging Face Auto 绫?    if model_is_on_hf_hub(cfg["config_path"]):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg["config_path"])
        # Overwrite VLA path
        cfg["config_path"] = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    processor = AutoProcessor.from_pretrained(cfg["config_path"], trust_remote_code=True)

    # 3. 鍔犺浇 VLM 妯″瀷 (浠?prism-qwen25-extra-dinosiglip-224px-0_5b)
    print(f"[*] Loading VLM model from: {cfg['vlm_path']}")
    # 浣犻渶瑕佷粠 prismatic.models 涓鍏?load 鍑芥暟
    from prismatic.models import load
    hf_token = ''
    if 'prism-qwen25-extra-dinosiglip-224px-0_5b' in cfg['vlm_path']:

        vlm = load(cfg["vlm_path"], hf_token=hf_token, load_for_training=True)
    else:
        vlm = load_vla(
            cfg['vlm_path'],
            hf_token=hf_token,
            load_for_training=True,
        )
    # 4. 鍔犺浇 VLA 閰嶇疆鏂囦欢 (纭紪鐮佽矾寰勪互涓庡弬鑰冧唬鐮佸畬鍏ㄤ竴鑷?
    print("[*] Loading VLA config from: <PATH_TO_VLA_CONFIG>/config.json")
    vla_config = AutoConfig.from_pretrained("<PATH_TO_VLA_CONFIG>/config.json")

    # 5. 鏍规嵁閰嶇疆鍒涘缓 VLA 妯″瀷 (姝ゆ椂鏉冮噸鏄殢鏈哄垵濮嬪寲鐨?
    print("[*] Creating VLA model from config...")
    vla = AutoModelForVision2Seq.from_config(vla_config, torch_dtype=torch.bfloat16).to(device)

    # 6. 瀹氫箟鏉冮噸閿悕鏇挎崲瑙勫垯 (涓庡弬鑰冧唬鐮佸畬鍏ㄧ浉鍚?
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

    # 7. 杞崲 VLM 鏉冮噸閿悕浠ュ尮閰?VLA 妯″瀷
    print("[*] Renaming VLM state dict keys...")
    old_state_dict = vlm.state_dict()
    RAW_STATE_DICT = rename_state_dict_keys(old_state_dict, replace_map)
    del old_state_dict  # 閲婃斁鍐呭瓨
    del vlm  # 閲婃斁 VLM 妯″瀷鍗犵敤鐨勫唴瀛?
    # 8. 灏嗚浆鎹㈠悗鐨勬潈閲嶅姞杞藉埌 VLA 妯″瀷涓?    print("[*] Loading converted weights into VLA model...")
    missing_keys, unexpected_keys = vla.load_state_dict(RAW_STATE_DICT, strict=False)
    print(f"    - Missing keys: {len(missing_keys)}")
    print(f"    - Unexpected keys: {len(unexpected_keys)}")

    # 9. 璁剧疆杈撳叆鍥惧儚鐨勬暟閲?(涓庡弬鑰冧唬鐮佷竴鑷?
    if "num_images_in_input" in cfg:
        print(f"[*] Setting number of images in input to: {cfg['num_images_in_input']}")
        vla.vision_backbone.set_num_images_in_input(cfg["num_images_in_input"])

    # --- 鏍稿績淇敼缁撴潫 ---

    # 鍚庣画閮ㄥ垎淇濇寔涓嶅彉锛屼絾闇€瑕佺‘淇?llm_dim 鐨勮幏鍙栨纭?    # 浠?VLA 妯″瀷涓幏鍙?llm_dim锛岃€屼笉鏄粠 config
    # 鍥犱负 vla 鍙兘鏄?DDP 鍖呰鐨勶紝鎵€浠ョ敤 .module
    llm_dim = vla.module.llm_dim if hasattr(vla, 'module') else vla.llm_dim
    print(f"[*] LLM dimension detected as: {llm_dim}")

    # 鍒涘缓鐙珛鐨?Action Head 鍜?Proprio Projector
    print("[*] Creating standalone action head and proprio projector...")
    if cfg["use_moe_action_head"]:
        action_head = StageMoEActionHead(
            input_dim=llm_dim,
            hidden_dim=llm_dim,
            action_dim=cfg.get("ACTION_DIM", 8),  # 浣跨敤 cfg 涓殑鍊硷紝榛樿 8
        ).to(device)
    else:
        action_head = L1RegressionActionHead(
            input_dim=llm_dim,
            hidden_dim=llm_dim,
            action_dim=cfg.get("ACTION_DIM", 8),
        ).to(device)

    proprio_projector = ProprioProjector(
        llm_dim=llm_dim,
        proprio_dim=cfg.get("proprio_dim", 8),  # 浠?cfg 鑾峰彇锛岄粯璁?8
    ).to(device)

    # 鍐荤粨 Action Head 鍜?Proprio Projector锛堝鏋滈厤缃簡锛?    if cfg.get("freeze_vla_action_head", False):
        print("[*] Freezing standalone action head and proprio projector.")
        for param in action_head.parameters():
            param.requires_grad = False
        for param in proprio_projector.parameters():
            param.requires_grad = False

    # 鍒涘缓 Discriminator (杩欓儴鍒嗘槸浣犵殑閫昏緫锛屼繚鎸佷笉鍙?
    print("[*] Creating Action ID Discriminator...")
    vision_backbone = DinoSigLIPViTBackbone(
        vision_backbone_id=cfg["vision_backbone_id"],
        image_resize_strategy=cfg["image_resize_strategy"],
        default_image_size=cfg["image_size"],
    ).to(device)

    if cfg.get("freeze_vision_backbone", False):
        for param in vision_backbone.parameters():
            param.requires_grad = False
        print("[*] Frozen DinoSigLIP vision backbone in discriminator.")

    discriminator = ActionIDDiscriminator(
        num_action_ids=cfg["num_action_ids"],
        vision_backbone=vision_backbone,
        llm_dim=llm_dim,
        proprio_dim=cfg.get("proprio_dim", 8),
        hidden_dim=cfg.get("discriminator_hidden_dim", 512)
    ).to(device)
    discriminator = discriminator.to(torch.bfloat16)
    print("[*] Models created successfully.\n")
    # 杩斿洖鎵€鏈夐渶瑕佺殑缁勪欢
    return vla, action_head, proprio_projector, discriminator, processor


def create_data_loaders(cfg: dict, processor: PrismaticProcessor) -> tuple[DataLoader, DataLoader]:
    """鍒涘缓骞惰繑鍥炶缁冨拰楠岃瘉鏁版嵁鍔犺浇鍣ㄣ€?""
    print("[*] Creating data loaders...")
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=True,
        use_proprio=True,
        use_minivlm=True
    )

    resize_resolution = (cfg["image_size"], cfg["image_size"])

    train_dataset = RLDSDataset(
        data_root_dir=cfg["data_root_dir"], data_mix=cfg["dataset_name"],
        batch_transform=batch_transform, resize_resolution=resize_resolution,
        shuffle_buffer_size=100_000, image_aug=True
    )

    collator = PaddedCollatorForActionPrediction(
        model_max_length=processor.tokenizer.model_max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
        padding_side="right"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    print(f"[*] Training dataset size: {len(train_dataset)}")
    print("[*] Validation uses the same train loader to avoid duplicating RLDS memory.\n")
    return train_loader, train_loader


def validate(cfg: dict, step: int, discriminator: nn.Module, vla: nn.Module, val_loader: DataLoader, loss_fn,num_patches,
             device: torch.device):
    """鍦ㄩ獙璇侀泦涓婅瘎浼版ā鍨嬫€ц兘銆?""
    print(f"\n[*] Running validation at step {step}...")
    discriminator.eval()
    vla.eval()  # VLA 濮嬬粓鍦ㄨ瘎浼版ā寮?
    all_metrics = {"loss": [], "accuracy": []}
    val_start_time = time.time()

    with torch.no_grad():
        for batch in val_loader:
            if time.time() - val_start_time > cfg["val_time_limit"]:
                print("[!] Validation time limit exceeded. Stopping early.")
                break

            # --- 浣犵殑鏍稿績楠岃瘉閫昏緫锛屼笌璁粌鏃跺畬鍏ㄧ浉鍚?---
            pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device)
            proprio = batch["proprio"].to(device)
            action_ids = batch["action_id"].squeeze().to(device)
            labels = batch.get("labels").to(device)
            input_ids = batch.get("input_ids").to(device)
            attention_mask = batch.get("attention_mask").to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                    output_hidden_states=True,
                    proprio=True,
                    proprio_projector=True,
                    noisy_actions=None,
                    noisy_action_projector=None,
                    diffusion_timestep_embeddings=None,
                    use_film=False
                )

            ground_truth_token_ids = batch["labels"][:, 1:].to(device)
            current_action_mask = get_current_action_mask(ground_truth_token_ids)
            next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

            # 鐗瑰緛鎻愬彇
            task_features = output.hidden_states[-1][:, :num_patches, :].to(torch.bfloat16)  # [B, num_patches, D]

            # Action鐗瑰緛鏄姩浣渢oken鐨勯殣钘忕姸鎬?            # 鎴戜滑闇€瑕佷粠鏂囨湰闅愯棌鐘舵€佷腑鎻愬彇鍔ㄤ綔鐩稿叧鐨勯儴鍒?            text_hidden_states = output.hidden_states[-1][:, num_patches:-1, :]  # [B, text_len, D]
            action_features = text_hidden_states[current_action_mask | next_actions_mask].reshape(
                cfg["batch_size"], -1, text_hidden_states.shape[-1]).to(torch.bfloat16)  # [B, num_action_tokens, D]

            # Discriminator 鍓嶅悜浼犳挱
            action_probs, confidence, logits = discriminator(
                pixel_values=pixel_values,
                proprio=proprio,
                input_ids=input_ids,
                task_features=task_features,
                action_features=action_features
            )

            loss = loss_fn(logits, action_ids, confidence)
            accuracy = (logits.argmax(dim=-1) == action_ids).float().mean()

            all_metrics["loss"].append(loss.item())
            all_metrics["accuracy"].append(accuracy.item())

    avg_loss = sum(all_metrics["loss"]) / len(all_metrics["loss"])
    avg_acc = sum(all_metrics["accuracy"]) / len(all_metrics["accuracy"])

    print(f"[*] Validation Results - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

    # 璁板綍鍒?wandb
    wandb.log({
        "val/loss": avg_loss,
        "val/accuracy": avg_acc,
    }, step=step)

    discriminator.train()  # 鍒囨崲鍥炶缁冩ā寮?    return avg_acc



def find_latest_checkpoint(save_dir: str) -> str:
    """鍦ㄤ繚瀛樼洰褰曚腑鏌ユ壘鏈€杩戠殑checkpoint鏂囦欢銆?""
    if not os.path.exists(save_dir):
        return None

    # 鏌ユ壘鎵€鏈塩heckpoint鏂囦欢
    checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith("checkpoint_step_") and f.endswith(".pth")]
    if not checkpoint_files:
        # 妫€鏌ユ槸鍚︽湁best_model
        best_model = os.path.join(save_dir, "best_model.pth")
        return best_model if os.path.exists(best_model) else None

    # 鎸夋楠ゅ彿鎺掑簭锛屽彇鏈€鏂扮殑
    def get_step_number(filename):
        return int(filename.split("_")[-1].split(".")[0])

    checkpoint_files.sort(key=get_step_number, reverse=True)
    return os.path.join(save_dir, checkpoint_files[0])

def load_checkpoint(cfg: dict, discriminator: nn.Module, optimizer: AdamW, scheduler: MultiStepLR,
                    device: torch.device) -> tuple[int, float, str | None]:
    """
    鍔犺浇checkpoint骞舵仮澶嶆ā鍨嬨€佷紭鍖栧櫒銆佽皟搴﹀櫒鐘舵€併€?    杩斿洖锛?璧峰姝ラ, 鏈€浣抽獙璇佸噯纭巼)
    """
    if not cfg.get("resume", False):
        print("[*] Resume disabled, starting training from scratch.")
        return 0, 0.0, None

    checkpoint_path = cfg.get("resume_checkpoint_path")

    # 濡傛灉鏈寚瀹氳矾寰勶紝鑷姩鏌ユ壘鏈€杩戠殑checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(cfg["save_dir"])
        if checkpoint_path is None:
            print("[*] No checkpoint found, starting training from scratch.")
            return 0, 0.0, None

    print(f"[*] Loading checkpoint from: {checkpoint_path}")

    # 鍔犺浇checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 鎭㈠鍒ゅ埆鍣ㄧ姸鎬?    if "discriminator_state_dict" in checkpoint:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        print("[*] Discriminator state loaded successfully.")
    else:
        print("[!] Warning: discriminator_state_dict not found in checkpoint.")

    # 鎭㈠浼樺寲鍣ㄧ姸鎬?    if "optimizer_state_dict" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("[*] Optimizer state loaded successfully.")
    else:
        print("[!] Warning: optimizer_state_dict not found in checkpoint.")

    # 鎭㈠璋冨害鍣ㄧ姸鎬?    if "scheduler_state_dict" in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("[*] Scheduler state loaded successfully.")
    else:
        print("[!] Warning: scheduler_state_dict not found in checkpoint.")

    # 鎭㈠璁粌姝ラ
    start_step = checkpoint.get("step", 0)
    print(f"[*] Resuming training from step {start_step}")

    # 鎭㈠鏈€浣抽獙璇佸噯纭巼
    best_val_acc = checkpoint.get("val_accuracy", 0.0)
    print(f"[*] Best validation accuracy from checkpoint: {best_val_acc:.4f}")

    # 鎭㈠wandb run id锛堝鏋滄湁锛?    wandb_id = checkpoint.get("wandb_run_id")

    return start_step, best_val_acc, wandb_id


# ==============================================================================
# 3. 涓昏缁冨嚱鏁?(Main Training Function)
#    璁粌娴佺▼鐨勬牳蹇冿紝鍖呭惈瀹屾暣鐨勮缁冨惊鐜?# ==============================================================================

def train(cfg: dict):
    device = setup_environment(cfg)

    # 鍒濆鍖栨墍鏈夋ā鍨嬬粍浠?    vla, action_head, proprio_projector, discriminator, processor = create_models(cfg, device)
    # import pdb;pdb.set_trace()

    num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    train_loader, val_loader = create_data_loaders(cfg, processor)

    # 浼樺寲鍣ㄥ彧閽堝 discriminator 鐨勫弬鏁?    optimizer = AdamW(
        filter(lambda p: p.requires_grad, discriminator.parameters()),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"]
    )
    scheduler = MultiStepLR(optimizer, milestones=[cfg["num_steps_before_decay"]], gamma=cfg["lr_decay_gamma"])
    loss_fn = ActionIDLoss()

    # 鍔犺浇checkpoint锛堝鏋滄湁锛?    start_step, best_val_acc, wandb_id = load_checkpoint(cfg, discriminator, optimizer, scheduler, device)

    # 鍒濆鍖杦andb锛堟敮鎸佺画璁級
    setup_wandb(cfg)

    # 璁粌鐘舵€?    # 灏?discriminator 璁句负璁粌妯″紡锛屽叾浠栨ā鍨嬩繚鎸佽瘎浼版ā寮?    discriminator.train()
    vla.eval()
    action_head.eval()
    proprio_projector.eval()

    print("[*] Starting training loop...")
    # 璋冩暣杩涘害鏉¤寖鍥?    progress_bar = tqdm(range(start_step, cfg["max_steps"]), desc="Training", initial=start_step)
    data_iter = iter(train_loader)

    for step in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device)
        proprio = batch["proprio"].to(device)
        action_ids = batch["action_id"].squeeze().to(device)
        labels = batch.get("labels").to(device)
        input_ids = batch.get("input_ids").to(device)
        attention_mask = batch.get("attention_mask").to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                output_hidden_states=True,
                proprio=True,
                proprio_projector=True,
                noisy_actions=None,
                noisy_action_projector=None,
                diffusion_timestep_embeddings=None,
                use_film=False
            )

        ground_truth_token_ids = batch["labels"][:, 1:].to(device)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

        # 鐗瑰緛鎻愬彇
        task_features = output.hidden_states[-1][:, :num_patches, :].to(torch.bfloat16)  # [B, num_patches, D]

        # Action鐗瑰緛鏄姩浣渢oken鐨勯殣钘忕姸鎬?        # 鎴戜滑闇€瑕佷粠鏂囨湰闅愯棌鐘舵€佷腑鎻愬彇鍔ㄤ綔鐩稿叧鐨勯儴鍒?        text_hidden_states = output.hidden_states[-1][:, num_patches:-1, :]  # [B, text_len, D]
        action_features = text_hidden_states[current_action_mask | next_actions_mask].reshape(
            cfg["batch_size"], -1, text_hidden_states.shape[-1]).to(torch.bfloat16)  # [B, num_action_tokens, D]

        # Discriminator 鍓嶅悜浼犳挱
        action_probs, confidence, logits = discriminator(
            pixel_values=pixel_values,
            proprio=proprio,
            input_ids=input_ids,
            task_features=task_features,
            action_features=action_features
        )

        # 鎹熷け璁＄畻
        loss = loss_fn(logits, action_ids, confidence)
        action_id_accuracy = (logits.argmax(dim=-1) == action_ids).float().mean()

        # 鍙嶅悜浼犳挱
        loss.backward()

        if (step + 1) % cfg["grad_accumulation_steps"] == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # 鏃ュ織璁板綍
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{action_id_accuracy.item():.4f}")

        if step % 10 == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/accuracy": action_id_accuracy.item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
            }, step=step)

        # 楠岃瘉涓庝繚瀛?        if step > 0 and step % cfg["val_freq"] == 0:
            val_acc = validate(cfg, step, discriminator, vla, val_loader, loss_fn,num_patches,
                               device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(cfg["save_dir"], "best_model.pth")
                # 淇濆瓨wandb run id浠ヤ究缁
                torch.save({
                    'step': step,
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_accuracy': best_val_acc,
                    'wandb_run_id': wandb.run.id if wandb.run else None,
                }, best_model_path)
                print(f"[*] Saved best discriminator model with accuracy {best_val_acc:.4f} to {best_model_path}")

        if step > 0 and step % cfg["save_freq"] == 0:
            checkpoint_path = os.path.join(cfg["save_dir"], f"checkpoint_step_{step}.pth")
            torch.save({
                'step': step,
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': best_val_acc,
                'wandb_run_id': wandb.run.id if wandb.run else None,
            }, checkpoint_path)
            print(f"[*] Saved checkpoint to {checkpoint_path}")

    print("\n[*] Training completed.")
    print(f"[*] Best validation accuracy: {best_val_acc:.4f}")
    wandb.finish()


# ==============================================================================
# 4. 鑴氭湰鍏ュ彛 (Script Entry Point)
# ==============================================================================

if __name__ == "__main__":
    train(CONFIG)

