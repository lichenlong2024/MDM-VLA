import argparse
import collections
from pathlib import Path


def require_tfds():
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise RuntimeError(
            "This script requires `tensorflow_datasets` in the active environment."
        ) from exc
    return tfds


def main():
    parser = argparse.ArgumentParser(
        description="Count per-step action_id distribution for an RLDS/TFDS dataset directory."
    )
    parser.add_argument(
        "--tfds-dir",
        type=Path,
        required=True,
        help="Path to a TFDS version directory, e.g. <PATH_TO_DATA_ROOT>/libero_bread_action_id/1.0.0",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to inspect. Default: train",
    )
    args = parser.parse_args()

    if not args.tfds_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.tfds_dir}")

    tfds = require_tfds()
    builder = tfds.builder_from_directory(str(args.tfds_dir))
    ds = builder.as_dataset(split=args.split)

    counter = collections.Counter()
    episode_lengths = []

    for episode in tfds.as_numpy(ds):
        step_count = 0
        for step in episode["steps"]:
            obs = step["observation"]
            action_id = int(obs["action_id"])
            counter[action_id] += 1
            step_count += 1
        episode_lengths.append(step_count)

    total_episodes = len(episode_lengths)
    total_steps = sum(episode_lengths)

    print(f"dataset_dir: {args.tfds_dir}")
    print(f"split: {args.split}")
    print(f"episodes: {total_episodes}")
    print(f"total_steps: {total_steps}")
    if total_episodes > 0:
        print(f"min_episode_len: {min(episode_lengths)}")
        print(f"max_episode_len: {max(episode_lengths)}")
        print(f"avg_episode_len: {total_steps / total_episodes:.2f}")

    print("\naction_id distribution:")
    for action_id in sorted(counter):
        count = counter[action_id]
        ratio = (count / total_steps) if total_steps > 0 else 0.0
        print(f"  action_id {action_id:>2}: count={count:<8} ratio={ratio:.4%}")


if __name__ == "__main__":
    main()

