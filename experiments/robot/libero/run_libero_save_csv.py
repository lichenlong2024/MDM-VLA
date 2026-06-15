"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union
import torch
import draccus
import numpy as np
import tqdm
from LIBERO.libero.libero import benchmark
import csv
import wandb
import numpy as np
import robosuite.utils.transform_utils as T


def save_trajectory_batch_to_csv(trajectory_records, output_path):
    """
    鎵归噺淇濆瓨杞ㄨ抗鏁版嵁鍒癈SV鏂囦欢

    Args:
        trajectory_records: 杞ㄨ抗璁板綍鍒楄〃锛屾瘡涓厓绱犳槸 {'x': ..., 'y': ..., 'z': ..., 'action_id': ...}
        output_path: 杈撳嚭CSV鏂囦欢璺緞
    """
    import os

    # 纭繚鐩綍瀛樺湪
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 鍐欏叆琛ㄥご
        writer.writerow(['x', 'y', 'z', 'action_id'])

        # 鍐欏叆鎵€鏈夋暟鎹?
        for record in trajectory_records:
            writer.writerow([record['x'], record['y'], record['z'], record['action_id']])

    #print(f"鉁?Saved {len(trajectory_records)} trajectory points to {output_path}")


# 鍦ㄩ€傚綋鐨勪綅缃皟鐢ㄦ鍑芥暟淇濆瓨杞ㄨ抗鏁版嵁
# 绀轰緥璋冪敤鏂瑰紡锛?
# save_trajectory_to_csv(ee_pos_data, f"trajectory_{task_name}_{trial_id}.csv", action_id)

def get_scene_info(obs, env):
    """
    浠庣幆澧冨璞′腑鎻愬彇鍦烘櫙鏍稿績淇℃伅锛岃繑鍥炴満姊拌噦銆乻ite銆佺墿浣撴湰浣撱€佺墿浣撳叧鑺傜殑缁撴瀯鍖栧瓧鍏搞€?

    Args:
        env: LIBERO/Robosuite鐜瀵硅薄锛堥渶鍖呭惈sim銆乷bs銆乺obots绛夋牳蹇冨睘鎬э級

    Returns:
        tuple: (robot_info, site_info, body_info, joint_info)
            - robot_info: 鏈烘鑷備俊鎭瓧鍏?
            - site_info: site淇℃伅瀛楀吀
            - body_info: 鐗╀綋鏈綋淇℃伅瀛楀吀
            - joint_info: 鐜涓墍鏈夊叧鑺傦紙鍚墿浣撳叧鑺傦級鐨勪俊鎭瓧鍏?
    """
    # -------------------------- 1. 鏈烘鑷備俊鎭?(robot_info) --------------------------
    robot_info = {
        "joint_states": None,  # 鍏宠妭浣嶇疆
        "gripper_states": None,  # 澶圭埅浣嶇疆
        "grasp_status": None,  # 鎶撳彇鐘舵€侊紙娉細闇€澶栭儴浼犲叆grasp锛屾澶勯鐣欙級
        "ee_pos": None,  # 鏈浣嶇疆锛堜笘鐣屽潗鏍囩郴锛?
        "ee_quat": None,  # 鏈濮挎€侊紙鍥涘厓鏁帮級
        "ee_states": None  # 鏈鐘舵€侊紙浣嶇疆+杞磋锛屼笌HDF5涓€鑷达級
    }

    # 浠巓bs鎻愬彇鏈烘鑷傛牳蹇冩暟鎹紙涓嶩DF5鏍煎紡瀵归綈锛?
    # 鍏宠妭浣嶇疆
    if "robot0_joint_pos" in obs:
        robot_info["joint_states"] = obs["robot0_joint_pos"].round(4).tolist()
    # 澶圭埅浣嶇疆
    if "robot0_gripper_qpos" in obs:
        robot_info["gripper_states"] = obs["robot0_gripper_qpos"].round(4).tolist()
    # 鏈鎵ц鍣ㄤ綅濮?
    if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
        ee_pos = obs["robot0_eef_pos"].round(4)
        ee_quat = obs["robot0_eef_quat"].round(4)
        ee_axisangle = T.quat2axisangle(ee_quat).round(4)
        ee_states = np.hstack((ee_pos, ee_axisangle)).tolist()
        robot_info["ee_pos"] = ee_pos.tolist()
        robot_info["ee_quat"] = ee_quat.tolist()
        robot_info["ee_states"] = ee_states

    # -------------------------- 2. Site淇℃伅 (site_info) --------------------------
    site_info = {}  # 閿細site鍚嶇О锛屽€硷細{ "xpos": 涓栫晫鍧愭爣绯讳綅缃?}
    sim = env.sim
    nsite = sim.model.nsite

    if nsite > 0:
        # 鑾峰彇鏈夋晥site鍚嶇О锛堣繃婊ょ┖瀛楃涓诧級
        site_names = [sim.model.site_id2name(i) for i in range(nsite) if sim.model.site_id2name(i)]
        # 鑾峰彇site瀹炴椂缁濆浣嶇疆锛堜笘鐣屽潗鏍囩郴锛?
        site_xpos = sim.data.site_xpos.round(4)  # shape: (nsite, 3)

        for name, xpos in zip(site_names, site_xpos):
            site_info[name] = {
                "xpos": xpos.tolist()  # [x, y, z]锛堝崟浣嶏細m锛?
            }

    # -------------------------- 3. 鐗╀綋鏈綋淇℃伅 (body_info) --------------------------
    body_info = {}  # 閿細body鍚嶇О锛屽€硷細{ "xpos": 浣嶇疆, "xquat": 濮挎€?}
    # 鑾峰彇鏈夋晥body鍚嶇О锛堣繃婊ょ┖瀛楃涓诧級
    all_body_names = [name for name in sim.model.body_names if name]

    # 鎻愬彇姣忎釜鐗╀綋鐨勫疄鏃朵綅濮匡紙涓栫晫鍧愭爣绯伙級
    for name in all_body_names:
        try:
            body_id = sim.model.body_name2id(name)
            xpos = sim.data.body_xpos[body_id].round(4)  # 浣嶇疆 [x,y,z]
            xquat = sim.data.body_xquat[body_id].round(4)  # 濮挎€?[qx,qy,qz,qw]
            body_info[name] = {
                "xpos": xpos.tolist(),
                "xquat": xquat.tolist()
            }
        except Exception as e:
            body_info[name] = {
                "xpos": None,
                "xquat": None,
                "error": f"鑾峰彇澶辫触: {str(e)[:20]}"
            }

    # -------------------------- 4. 鐗╀綋鍏宠妭淇℃伅 (joint_info) --------------------------
    joint_info = {}  # 閿細鍏宠妭鍚嶇О锛屽€硷細{ "angle": 鍏宠妭瑙掑害锛堝姬搴︼級 }
    njoint = sim.model.njnt  # 鎬诲叧鑺傛暟閲忥紙鍚満鍣ㄤ汉鍜岀幆澧冪墿浣擄級
    # 鑾峰彇鎵€鏈夊叧鑺傚悕绉帮紙杩囨护绌哄瓧绗︿覆锛?
    joint_names = [sim.model.joint_id2name(i) for i in range(njoint) if sim.model.joint_id2name(i)]
    for name in joint_names:
        # 鑾峰彇鍏宠妭ID
        joint_id = sim.model.joint_name2id(name)
        # 鑾峰彇鍏宠妭鍦╭pos涓殑绱㈠紩鑼冨洿锛坔inge鍏宠妭閫氬父鍙崰1涓储寮曪級
        adr = sim.model.jnt_qposadr[joint_id]
        # 鎻愬彇鍏宠妭瑙掑害锛坬pos涓瓨鍌ㄧ殑鏄叧鑺備綅缃紝瀵筯inge鍏宠妭鑰岃█灏辨槸瑙掑害锛?
        angle = sim.data.qpos[adr].round(4).item()  # .item()杞负鏍囬噺
        joint_info[name] = {
            "angle": angle  # 鍗曚綅锛氬姬搴?
        }

    return robot_info, site_info, body_info, joint_info


# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
    get_action_with_action_id,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

# Import for MoE action head and Action ID Discriminator
from prismatic.models.action_heads import StageMoEActionHead
from prismatic.models.action_id_discriminator import ActionIDDiscriminator
from prismatic.models.backbones.vision.dinosiglip_vit import DinoSigLIPViTBackbone
from prismatic.vla.action_tokenizer import ActionTokenizer


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"  # Model family
    pretrained_checkpoint: Union[str, Path] = ""  # Pretrained checkpoint path
    use_l1_regression: bool = True  # If True, uses continuous action head with L1 regression objective
    use_minivlm: bool = True  # If True, uses minivlm
    num_diffusion_steps: int = 50  # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False  # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2  # Number of images in the VLA input (default: 1)
    use_proprio: bool = True  # Whether to include proprio state in input

    center_crop: bool = True  # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8  # Number of actions to execute open-loop before requerying policy
    unnorm_key: Union[str, Path] = ""  # Action un-normalization key

    load_in_8bit: bool = False  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False  # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    initial_states_path: str = "DEFAULT"  # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256  # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None  # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"  # Local directory for eval logs

    use_wandb: bool = False  # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"  # Name of WandB entity
    wandb_project: str = "your-wandb-project"  # Name of WandB project

    seed: int = 7  # Random Seed (for reproducibility)

    # fmt: on
    save_version: str = "vla-adapter"  # version of
    use_pro_version: bool = True  # encourage to use the pro models we released.
    phase: str = "Inference"

    # MoE and Action ID Discriminator parameters
    use_moe_action_head: bool = False  # If True, uses MoE action head instead of L1 regression head
    use_action_id_discriminator: bool = False  # If True, uses Action ID discriminator
    stage_definitions: str = "0:0-2,1:3-10,2:11-17"  # Stage definitions in format "stage_id:start-end,stage_id:start-end,..."
    num_action_ids: int = 18  # Number of action IDs in the dataset
    top_k: int = 3  # Number of top experts to use in MoE


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def parse_stage_definitions(stage_definitions_str):
    """Parse stage definition string"""
    stage_definitions = {}
    for stage_def in stage_definitions_str.split(','):
        stage_id, action_range = stage_def.split(':')
        start, end = map(int, action_range.split('-'))
        stage_definitions[int(stage_id)] = list(range(start, end + 1))
    return stage_definitions


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)
    model.set_version(cfg.save_version)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression:
        if cfg.use_moe_action_head:
            # Use MoE action head
            stage_definitions = parse_stage_definitions(cfg.stage_definitions)
            action_head = StageMoEActionHead(
                input_dim=model.llm_dim,
                hidden_dim=model.llm_dim,
                action_dim=7,  # ACTION_DIM
                use_pro_version=cfg.use_pro_version,
                stage_definitions=stage_definitions,
                num_action_ids=cfg.num_action_ids,
                top_k=cfg.top_k
            )

            # Load action head weights
            from experiments.robot.openvla_utils import find_checkpoint_file, load_component_state_dict
            checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "action_head")
            state_dict = load_component_state_dict(checkpoint_path)
            action_head.load_state_dict(state_dict)
            action_head = action_head.to(torch.bfloat16).to("cuda:0" if torch.cuda.is_available() else "cpu")
            action_head.eval()
        else:
            # Use standard L1 regression action head
            action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    # Initialize Action ID Discriminator if needed
    action_id_discriminator = None
    if cfg.use_action_id_discriminator:
        vision_backbone = DinoSigLIPViTBackbone(
            vision_backbone_id="dinosiglip-vit-so-224px",
            image_resize_strategy="resize-naive",
        )
        action_id_discriminator = ActionIDDiscriminator(
            num_action_ids=cfg.num_action_ids,
            vision_backbone=vision_backbone,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
            llm_dim=model.llm_dim
        )

        # Load action ID discriminator weights
        from experiments.robot.openvla_utils import find_checkpoint_file, load_component_state_dict
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "action_id_discriminator")
        state_dict = load_component_state_dict(checkpoint_path)
        action_id_discriminator.load_state_dict(state_dict)
        action_id_discriminator = action_id_discriminator.to(torch.bfloat16).to(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        action_id_discriminator.eval()

    return model, action_head, proprio_projector, noisy_action_projector, processor, action_id_discriminator


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    # unnorm_key = cfg.task_suite_name
    unnorm_key = 'libero_10_action_id'
    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def run_episode(
        cfg: GenerateConfig,
        env,
        task_description: str,
        model,
        resize_size,
        processor=None,
        action_head=None,
        proprio_projector=None,
        noisy_action_projector=None,
        action_id_discriminator=None,
        initial_state=None,
        log_file=None,
        episode_id =None
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize actiosn queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              "{NUM_ACTIONS_CHUNK} constant defined in prismatic.vla.constants! For best performance (in terms of "
              "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    # try:
    # 鍒濆鍖栬建杩硅褰?
    trajectory_records = []  # 鐢ㄥ垪琛ㄦ敹闆嗘墍鏈夋暟鎹偣
    while t < max_steps + cfg.num_steps_wait:
        # Do nothing for the first few timesteps to let objects stabilize
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue
        # import pdb;pdb.set_trace()
        # Prepare observation
        observation, img = prepare_observation(obs, resize_size)
        replay_images.append(img)

        # If action queue is empty, requery model
        if len(action_queue) == 0:
            # Query model to get action

            actions, action_id_pred = get_action_with_action_id(
                cfg,
                model,
                observation,
                task_description,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                action_id_discriminator=action_id_discriminator,
                use_moe_action_head=cfg.use_moe_action_head,
                use_film=cfg.use_film,
                use_minivlm=cfg.use_minivlm
            )
            # log_message(f"\naction: {actions}", log_file)
            action_queue.extend(actions)

        # Get action from queue
        action = action_queue.popleft()
        # action = actions[0]

        # Process action
        action = process_action(action, cfg.model_family)

        # Execute action in environment
        obs, reward, done, info = env.step(action.tolist())
        robot_info, site_info, body_info, joint_info = get_scene_info(obs, env)
        ee_pos = robot_info["ee_pos"]
        #(f"鏈烘鑷傛湯绔綅缃? {ee_pos}, 妯″瀷棰勬祴鍔ㄤ綔鍩哄厓: {action_id_pred}")
        trajectory_records.append({
            'x': ee_pos[0],
            'y': ee_pos[1],
            'z': ee_pos[2],
            'action_id': float(action_id_pred)
        })
        csv_filename = f"<PATH_TO_TRAJECTORY_OUTPUT_DIR>/trajectory_task_{task_description}_trial_{episode_id}.csv"
        save_trajectory_batch_to_csv(trajectory_records, csv_filename)
        if done:
            success = True
            break
        t += 1

    # except Exception as e:
    #     log_message(f"Episode error: {e}", log_file)
    return success, replay_images


def run_task(
        cfg: GenerateConfig,
        task_suite,
        task_id: int,
        model,
        resize_size,
        processor=None,
        action_head=None,
        proprio_projector=None,
        noisy_action_projector=None,
        action_id_discriminator=None,
        total_episodes=0,
        total_successes=0,
        log_file=None,
        save_version=None
):
    """Run evaluation for a single task."""
    # Get task
    # task_id = 8
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            action_id_discriminator,
            initial_state,
            log_file,
            episode_idx,
        )
        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file,
            save_version=save_version
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # close env
    env.close()
    del env

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor, action_id_discriminator = initialize_model(
        cfg)

    # for name, param in model.named_parameters():
    #     if 'action_queries' in name:
    #         print(f"{name}: {param}")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            action_id_discriminator,
            total_episodes,
            total_successes,
            log_file,
            cfg.save_version
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
