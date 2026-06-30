# Action-ID Schema

This document describes the default action-ID schema used by MDM-VLA. The labels are mid-level semantic annotations for routing and discriminator supervision. They are not appended to the continuous action target.

## Storage Format

For RLDS/TFDS-style datasets, the action ID is stored at each timestep as:

```text
steps["observation"]["action_id"]
```

The field is an integer scalar in `[0, 17]`. During policy training, the continuous target remains the original action chunk. The action ID is used to supervise the multimodal action-ID discriminator and to derive stage-level routing priors for the sparse MoE action head.

## Default Action IDs

| ID | Name | Short Meaning |
|---:|---|---|
| 0 | approach | Move the arm toward the object before contact or grasping. |
| 1 | posture_adjustment | Adjust end-effector posture after approach and before interaction. |
| 2 | retract | Move the arm away from the object after task completion. |
| 3 | pull | Grasp or contact an object and drag it toward the arm direction. |
| 4 | push | Contact an object and push it away or along the support surface. |
| 5 | pick | Grasp and lift an object from its support surface. |
| 6 | place | Carry an object and place it at a target location. |
| 7 | pour | Tilt a carried container-like object to pour its contents. |
| 8 | twist | Twist a grasped object around its own axis. |
| 9 | slide | Maintain contact and slide an object along the support surface. |
| 10 | other_move | Other object-interaction motions not covered above. |
| 11 | align | Adjust the relative pose between a carried object and the target. |
| 12 | lift | Make a small vertical upward adjustment while carrying an object. |
| 13 | lower | Make a small vertical downward adjustment while carrying an object. |
| 14 | tilt | Adjust the tilt angle of the gripper or carried object. |
| 15 | rotate | Rotate a carried object around a vertical or task-relevant axis. |
| 16 | flip | Flip a carried object around a horizontal axis. |
| 17 | press | Apply a short vertical pressure on an object or surface. |

## Default Semantic Stages

The default paper setting groups the 18 action IDs into three semantic stages:

```text
0:0-2,1:3-10,2:11-17
```

| Stage | Action IDs | Interpretation |
|---:|---|---|
| S1 | 0-2 | Free-space or arm-only motion before/after object interaction. |
| S2 | 3-10 | Gripper-object interaction and object transport. |
| S3 | 11-17 | Fine adjustment, alignment, rotation, and contact refinement. |

These stage definitions are configurable. They are intended as mid-level semantic groupings rather than universal closed-loop skills with fixed dynamics.

## Adding Action IDs to an RLDS Dataset

The helper script `scripts/rewrite_rlds_add_action_id.py` can be used when a task has known step ranges for each local motion regime.

Example:

```bash
python scripts/rewrite_rlds_add_action_id.py \
  --input-dir /path/to/tensorflow_datasets/libero_task/1.0.0 \
  --output-dir /path/to/tensorflow_datasets/libero_task_action_id/1.0.0 \
  --dataset-name libero_task_action_id \
  --range 0-35:0 \
  --range 35-75:5 \
  --range 75-end:15
```

The example assigns:

| Step Range | Action ID |
|---|---:|
| 0-34 | 0 |
| 35-74 | 5 |
| 75-end | 15 |

For new tasks, inspect demonstrations and define task-specific step ranges or annotation rules before rewriting the dataset. The rewritten dataset should expose `observation.action_id` to the dataset loader.

## Recommended Validation

After annotation, verify:

| Check | Expected Result |
|---|---|
| `action_id` exists in every timestep | Yes |
| IDs lie in `[0, 17]` | Yes |
| stage mapping covers every used ID | Yes |
| boundary locations match the intended local motion regimes | Manually inspect a few trajectories |

