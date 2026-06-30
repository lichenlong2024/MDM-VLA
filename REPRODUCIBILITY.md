# Reproducibility Guide

This guide describes how to reproduce the MDM-VLA training and evaluation pipeline. It is written for reviewers and researchers who want to inspect the method implementation or reproduce the reported results with their own local datasets and checkpoints.

## Reproduction Levels

| Level | What Can Be Reproduced | Required External Assets |
|---|---|---|
| Code inspection | Model architecture, routing logic, discriminator, MoE action head | None beyond this repository. |
| LIBERO training from scratch | Discriminator training, MDM-VLA fine-tuning, LIBERO evaluation | Pretrained VLM, LIBERO data, action-ID annotations. |
| Checkpoint evaluation | LIBERO success rates using trained checkpoints | Released or locally trained checkpoints. |
| Real-robot evaluation | Franka task execution | Matching hardware, task data, checkpoints, and local robot controller. |

## 1. Environment Setup

Create a Python environment and install the package:

```bash
conda create -n mdm-vla python=3.10 -y
conda activate mdm-vla
pip install -e .
```

The full development environment used during our experiments is listed in `requirements_full.txt`. Some dependencies are platform-specific and may require manual installation:

| Dependency | Notes |
|---|---|
| PyTorch/CUDA | Match the local GPU and CUDA driver. |
| FlashAttention | Install a wheel compatible with the local CUDA and PyTorch version. |
| TensorFlow / TFDS | Required for RLDS-style datasets. |
| LIBERO | Required for simulation evaluation. |
| MuJoCo / robosuite | Required by LIBERO. |

## 2. Prepare Datasets

Prepare the original LIBERO or real-robot RLDS datasets under a local data root:

```text
<DATA_ROOT>/
  libero_spatial_action_id/
  libero_object_action_id/
  libero_goal_action_id/
  libero_10_action_id/
  real_robot_task_action_id/
```

MDM-VLA expects each timestep to contain:

```text
steps["observation"]["action_id"]
```

See `ACTION_ID_SCHEMA.md` for the default 18 action IDs and three-stage grouping.

For tasks with known step ranges, rewrite an RLDS dataset with:

```bash
python scripts/rewrite_rlds_add_action_id.py \
  --input-dir /path/to/tensorflow_datasets/libero_task/1.0.0 \
  --output-dir /path/to/tensorflow_datasets/libero_task_action_id/1.0.0 \
  --dataset-name libero_task_action_id \
  --range 0-35:0 \
  --range 35-75:5 \
  --range 75-end:15
```

For full reproduction, use the same action-ID definitions and stage mapping as the paper:

```text
--num_action_ids 18
--stage_definitions "0:0-2,1:3-10,2:11-17"
```

## 3. Train the Multimodal Action-ID Discriminator

Edit the paths in `vla-scripts/train_action_id_discriminator.py` or pass equivalent configuration values in your local launcher. The key settings are:

| Setting | Paper-Style Value |
|---|---|
| Number of action IDs | `18` |
| Learning rate | `1e-4` |
| Max steps | `10000` for real-robot tasks; adjust for larger LIBERO subsets |
| Batch size | Hardware-dependent; keep effective batch size fixed when comparing methods |
| Visual backbone | DINO-SigLIP-style visual features used by the VLM backbone |

Example:

```bash
python vla-scripts/train_action_id_discriminator.py
```

The script saves checkpoints such as:

```text
checkpoint_step_<STEP>.pth
```

The discriminator checkpoint is later loaded during policy fine-tuning and frozen in the protocol-matched MDM-VLA setting.

## 4. Fine-Tune the MDM-VLA Policy

Use `vla-scripts/finetune_train_by_self_weight.py` for the discriminator-guided MoE policy. The paper setting uses two image inputs, proprioceptive conditioning, image augmentation, LoRA fine-tuning, continuous L1 action regression, and action chunks of length 8.

Example LIBERO-style command:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/finetune_train_by_self_weight.py \
  --vlm_path /path/to/pretrained_vlm \
  --config_file_path /path/to/pretrained_vlm/configs \
  --data_root_dir /path/to/data_root \
  --dataset_name libero_10_action_id \
  --run_root_dir /path/to/output_dir \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --num_steps_before_decay 100000 \
  --max_steps 100000 \
  --save_freq 20000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --lora_rank 64 \
  --use_action_id_discriminator True \
  --train_action_id_discriminator False \
  --action_id_discriminator_path /path/to/discriminator_checkpoint.pth \
  --num_action_ids 18 \
  --use_moe_action_head True \
  --stage_definitions "0:0-2,1:3-10,2:11-17"
```

For real-robot task-specific fine-tuning, use the same input and optimization settings but train for `10000` steps:

```text
--max_steps 10000
--num_steps_before_decay 10000
```

## 5. Train a Protocol-Matched Baseline

To train a baseline without action-ID routing and sparse expert specialization, disable the discriminator and MoE action head:

```text
--use_action_id_discriminator False
--use_moe_action_head False
```

Keep the same VLM inputs, data, LoRA settings, learning rate, effective batch size, and training steps as MDM-VLA.

## 6. Evaluate on LIBERO

Use the LIBERO evaluation entry point:

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /path/to/policy_checkpoint_dir \
  --task_suite_name libero_10 \
  --num_trials_per_task 50 \
  --num_open_loop_steps 8 \
  --use_l1_regression True \
  --use_proprio True \
  --use_moe_action_head True \
  --use_action_id_discriminator True \
  --num_action_ids 18 \
  --stage_definitions "0:0-2,1:3-10,2:11-17"
```

Run the same script for the four LIBERO suites used in the paper:

```text
libero_spatial
libero_object
libero_goal
libero_10
```

Report task success rate as the evaluation metric. For protocol-matched comparisons, keep the same number of trials, initial-state protocol, training steps, and model inputs across the baseline and MDM-VLA.

## 7. Standalone Discriminator Evaluation

Use:

```bash
python vla-scripts/val_action_id_dis.py
```

Recommended metrics:

| Metric | Purpose |
|---|---|
| Top-1 action-ID accuracy | Exact action-ID prediction. |
| Stage-level accuracy | Routing-level correctness after grouping IDs into stages. |
| Boundary-tolerant accuracy | Ignores small errors near action transitions. |
| Cross-stage error rate | Measures errors that would select the wrong expert stage. |

Stage-level and cross-stage metrics are especially important because MDM-VLA routes sparse experts at the semantic-stage level.

## 8. Real-Robot Evaluation

The real-robot policy uses the same model path with task-specific checkpoints. A deployment server is provided under:

```text
experiments/robot/server_deploy/
```

Typical server-side command:

```bash
python experiments/robot/server_deploy/franka_deploy_json.py \
  --host 127.0.0.1 \
  --port 8000 \
  --pretrained_checkpoint /path/to/policy_checkpoint_dir \
  --num_images_in_input 2 \
  --use_proprio true \
  --proprio_dim 8 \
  --use_pro_version true \
  --use_moe_action_head true \
  --use_action_id_discriminator true \
  --stage_definitions "0:0-2,1:3-10,2:11-17" \
  --num_action_ids 18
```

Real-robot reproduction requires a compatible robot arm, gripper, cameras, controller, action scaling, and safety limits. The released utilities cover the model-side inference path, but users must adapt the low-level robot control stack to their own hardware.

## 9. Common Reproduction Checks

Before comparing results, verify:

| Check | Expected Result |
|---|---|
| Dataset contains `observation.action_id` | Required for discriminator and MDM-VLA training. |
| `num_action_ids` matches the dataset | Default is `18`. |
| `stage_definitions` covers all used action IDs | Default is `0:0-2,1:3-10,2:11-17`. |
| Discriminator is frozen during policy fine-tuning | Use `--train_action_id_discriminator False`. |
| MDM-VLA uses MoE action head | Use `--use_moe_action_head True`. |
| Baseline removes routing and MoE | Use `--use_action_id_discriminator False --use_moe_action_head False`. |
| Action chunk length matches evaluation | Default is `8`. |
| Dataset statistics are present in checkpoint directory | Required for action unnormalization. |

## 10. Known Limitations of This Code Release

This compact release does not include paper checkpoints or annotated data. Therefore, exact numerical reproduction requires either:

1. Running the full training pipeline from locally prepared annotated datasets, or
2. Using released datasets/checkpoints when they become available.

For real-robot experiments, exact reproduction also depends on hardware, calibration, low-level control, and safety settings.

