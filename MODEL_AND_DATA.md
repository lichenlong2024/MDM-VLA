# Model, Data, and Checkpoint Availability

This repository is a compact code release for the core MDM-VLA method. It intentionally does not include large datasets, pretrained backbones, training logs, videos, or model checkpoints.

## Included in This Repository

| Component | Included | Notes |
|---|---:|---|
| MDM-VLA model code | Yes | Discriminator-guided sparse MoE action routing. |
| Multimodal action-ID discriminator | Yes | `prismatic/models/action_id_discriminator.py`. |
| Sparse stage-level MoE action head | Yes | `prismatic/models/action_heads.py`. |
| LIBERO training and evaluation scripts | Yes | Entry points are under `vla-scripts/` and `experiments/robot/libero/`. |
| Real-robot deployment utilities | Yes | Server-side utilities are under `experiments/robot/server_deploy/`. |
| Action-ID annotation utilities | Yes | `scripts/rewrite_rlds_add_action_id.py`. |
| Annotated datasets | No | To be prepared separately or released after publication. |
| Pretrained VLM weights | No | Download or prepare separately. |
| Policy checkpoints | No | Train from scratch or use released checkpoints when available. |
| Discriminator checkpoints | No | Train from scratch or use released checkpoints when available. |
| Real-robot raw data | No | Not included in this compact code release. |

## External Assets Needed for Full Reproduction

To reproduce the paper experiments from scratch, prepare the following resources:

| Asset | Required For | Expected Form |
|---|---|---|
| Pretrained VLM backbone | Discriminator training and policy fine-tuning | Local checkpoint directory, e.g. `/path/to/pretrained_vlm`. |
| LIBERO datasets | Simulation training and evaluation | TFDS/RLDS-compatible dataset root. |
| Action-ID-annotated LIBERO datasets | MDM-VLA training | RLDS datasets with `steps["observation"]["action_id"]`. |
| Discriminator checkpoint | Frozen routing during policy fine-tuning | `.pth` checkpoint produced by `vla-scripts/train_action_id_discriminator.py`. |
| Fine-tuned policy checkpoint | Evaluation and deployment | Checkpoint directory containing VLM/LoRA components, action head, and statistics. |
| Real-robot demonstrations | Real-robot training | Task-specific RLDS-style data with action IDs. |

## Expected Checkpoint Directory Structure

Evaluation and deployment scripts expect a checkpoint directory containing component files that can be found by name. A typical directory should include:

```text
<CHECKPOINT_DIR>/
  dataset_statistics.json
  action_head--<checkpoint_name>
  proprio_projector--<checkpoint_name>        # if proprioception is used
  action_id_discriminator--<checkpoint_name>  # for MDM-VLA
  <VLM or LoRA checkpoint files>
```

The exact filenames may vary with the training script and checkpoint suffix. The loader searches for files containing component names such as `action_head`, `proprio_projector`, and `action_id_discriminator`.

## Data Release Note

The paper states that code and annotated data will be released upon publication. This repository currently provides the code and annotation utilities. If annotated data or pretrained checkpoints are not yet available, reproduce the method by following `REPRODUCIBILITY.md` and preparing local RLDS datasets with the schema in `ACTION_ID_SCHEMA.md`.

## Real-Robot Reproduction Scope

The real-robot experiments require a matching or adapted hardware stack. The code release includes server-side deployment utilities, but exact reproduction depends on:

| Requirement | Example in the Paper Setting |
|---|---|
| Robot arm | Franka Emika 7-DoF arm |
| Gripper | ZhiXingGripper or compatible end-effector |
| Cameras | Multi-view RGB observations, e.g. Intel RealSense D435i |
| Controller | Local robot controller and motion execution stack |
| Network setup | Optional inference server and client connection |

For different hardware, users should adapt state encoding, action scaling, camera preprocessing, and low-level execution.

