# MDM-VLA

This repository contains the core implementation of **MDM-VLA: Multimodal Discriminator-Guided Sparse Mixture-of-Experts for Long-Horizon Vision-Language-Action Control**.

The release is prepared for anonymous/reviewer inspection of the method implementation. It includes the model code, multimodal discriminator, sparse MoE action head, training scripts, discriminator validation scripts, LIBERO evaluation scripts, and real-robot deployment utilities. Datasets, checkpoints, pretrained backbone weights, logs, and paper source files are intentionally not included.

## Reproducibility Documents

- `REPRODUCIBILITY.md`: end-to-end reproduction guide for dataset preparation, discriminator training, policy fine-tuning, LIBERO evaluation, and real-robot deployment.
- `ACTION_ID_SCHEMA.md`: default 18 action IDs, three-stage grouping, and RLDS annotation format.
- `MODEL_AND_DATA.md`: what is included in this compact code release, what external assets are required, and the expected checkpoint structure.

## Core Components

- `prismatic/models/action_id_discriminator.py`: multimodal action-ID discriminator with visual, proprioceptive, task, and action-history inputs.
- `prismatic/models/action_heads.py`: sparse stage-level MoE action head and continuous action heads.
- `prismatic/extern/hf/modeling_prismatic.py`: VLA forward/prediction path with discriminator-guided action routing.
- `prismatic/vla/datasets/datasets.py`: RLDS/LIBERO dataset interface with action-ID supervision.
- `vla-scripts/train_action_id_discriminator.py`: discriminator training entry point.
- `vla-scripts/val_action_id_dis.py`: standalone discriminator evaluation and routing-signal analysis.
- `vla-scripts/finetune_office.py`: LIBERO policy fine-tuning entry point.
- `vla-scripts/finetune_train_by_self_weight.py`: real-robot/task-specific fine-tuning entry point.
- `experiments/robot/libero/run_libero_eval.py`: LIBERO evaluation script.
- `experiments/robot/server_deploy/`: real-robot inference/deployment utilities.
- `scripts/rewrite_rlds_add_action_id.py`: utility for adding action-ID labels to RLDS-style data.

## What Is Not Included

This repository does not include:

- LIBERO or real-robot datasets.
- Model checkpoints, LoRA weights, discriminator checkpoints, or pretrained VLM weights.
- W&B logs, generated figures, videos, paper source files, or local experiment records.

Please prepare datasets and pretrained backbones separately, then update paths in the training scripts according to your local environment.

## Installation

```bash
conda create -n mdm-vla python=3.10 -y
conda activate mdm-vla
pip install -e .
```

The full development environment used in our experiments is provided in `requirements_full.txt`. Some dependencies, such as FlashAttention, TensorFlow/RLDS utilities, LIBERO, and CUDA-specific packages, may require platform-specific installation.

## Training and Evaluation Entry Points

Train the multimodal discriminator:

```bash
python vla-scripts/train_action_id_discriminator.py
```

Validate the discriminator:

```bash
python vla-scripts/val_action_id_dis.py
```

Fine-tune on LIBERO:

```bash
python vla-scripts/finetune_office.py
```

Evaluate on LIBERO:

```bash
python experiments/robot/libero/run_libero_eval.py
```

Real-robot deployment utilities are under `experiments/robot/server_deploy/`.

## Notes for Reviewers

The code is released as a compact method implementation package. Paths in example commands should be replaced with local paths before running. The method-specific logic is concentrated in the discriminator, stage routing, and sparse MoE action head files listed above.
