import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import tensorflow as tf


def parse_ranges(range_specs: List[str]) -> List[Tuple[int, int, int]]:
    parsed = []
    for spec in range_specs:
        span, action_id_str = spec.split(":")
        action_id = int(action_id_str.strip())
        start_str, end_str = span.split("-")
        start = int(start_str.strip())
        end_token = end_str.strip().lower()
        end = -1 if end_token in {"end", "last", "rest"} else int(end_token)
        parsed.append((start, end, action_id))
    parsed.sort(key=lambda x: x[0])
    return parsed


def action_id_for_step(step_idx: int, rules: List[Tuple[int, int, int]]) -> int:
    for start, end, action_id in rules:
        if end == -1:
            if step_idx >= start:
                return action_id
        elif start <= step_idx < end:
            return action_id
    raise ValueError(f"No action_id rule matched step {step_idx}.")


def load_features(version_dir: Path) -> Dict:
    with open(version_dir / "features.json", "r", encoding="utf-8") as f:
        return json.load(f)


def save_features(version_dir: Path, features: Dict) -> None:
    with open(version_dir / "features.json", "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=4)


def ensure_action_id_feature(features: Dict) -> Dict:
    out = json.loads(json.dumps(features))
    obs_features = out["featuresDict"]["features"]["steps"]["sequence"]["feature"]["featuresDict"]["features"]["observation"][
        "featuresDict"
    ]["features"]
    if "action_id" not in obs_features:
        obs_features["action_id"] = {
            "description": "Action ID added for VLA-Adapter action discriminator training.",
            "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
            "tensor": {
                "dtype": "int64",
                "encoding": "none",
                "shape": {},
            },
        }
    return out


def build_signature() -> Dict[str, tf.TensorSpec]:
    return {
        "episode_metadata": {
            "file_path": tf.TensorSpec(shape=(), dtype=tf.string),
        },
        "steps": {
            "action": tf.TensorSpec(shape=(None, 7), dtype=tf.float32),
            "discount": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "is_first": tf.TensorSpec(shape=(None,), dtype=tf.bool),
            "is_last": tf.TensorSpec(shape=(None,), dtype=tf.bool),
            "is_terminal": tf.TensorSpec(shape=(None,), dtype=tf.bool),
            "language_instruction": tf.TensorSpec(shape=(None,), dtype=tf.string),
            "reward": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "observation": {
                "image": tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.uint8),
                "wrist_image": tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.uint8),
                "state": tf.TensorSpec(shape=(None, 8), dtype=tf.float32),
                "joint_state": tf.TensorSpec(shape=(None, 7), dtype=tf.float32),
                "action_id": tf.TensorSpec(shape=(None,), dtype=tf.int64),
            },
        },
    }


def dataset_generator(builder, split: str, rules: List[Tuple[int, int, int]]) -> Iterable[Dict]:
    ds = builder.as_dataset(split=split)
    for episode in tf.data.Dataset.as_numpy_iterator(ds):
        steps = episode["steps"]
        num_steps = len(steps["action"])
        action_ids = np.asarray([action_id_for_step(i, rules) for i in range(num_steps)], dtype=np.int64)

        yield {
            "episode_metadata": {
                "file_path": episode["episode_metadata"]["file_path"],
            },
            "steps": {
                "action": steps["action"].astype(np.float32),
                "discount": steps["discount"].astype(np.float32),
                "is_first": steps["is_first"].astype(bool),
                "is_last": steps["is_last"].astype(bool),
                "is_terminal": steps["is_terminal"].astype(bool),
                "language_instruction": steps["language_instruction"],
                "reward": steps["reward"].astype(np.float32),
                "observation": {
                    "image": steps["observation"]["image"].astype(np.uint8),
                    "wrist_image": steps["observation"]["wrist_image"].astype(np.uint8),
                    "state": steps["observation"]["state"].astype(np.float32),
                    "joint_state": steps["observation"]["joint_state"].astype(np.float32),
                    "action_id": action_ids,
                },
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite an RLDS dataset by adding action_id to every step.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Path to TFDS dataset version directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output TFDS dataset version directory.")
    parser.add_argument("--dataset-name", type=str, default="libero_bend_action_id", help="New TFDS dataset name.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to rewrite.")
    parser.add_argument(
        "--range",
        action="append",
        required=True,
        help="Step range spec in the form start-end:action_id, e.g. 0-35:0 or 75-end:15",
    )
    args = parser.parse_args()

    rules = parse_ranges(args.range)
    version_dir = args.input_dir
    parent_dir = version_dir.parent

    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError(
            "tensorflow_datasets is required to rewrite RLDS datasets. "
            "Please install it in the current environment."
        ) from exc

    builder = tfds.builder_from_directory(str(version_dir))
    signature = build_signature()
    rewritten = tf.data.Dataset.from_generator(
        lambda: dataset_generator(builder, args.split, rules),
        output_signature=signature,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_prefix = args.output_dir / f"{args.dataset_name}-{args.split}"
    written = tf.data.experimental.save(
        rewritten,
        str(args.output_dir / "tmp_tf_save"),
    )
    del written

    writer = tf.io.TFRecordWriter  # keep reference for clarity
    del writer

    tfds.folder_dataset.write_metadata(
        data_dir=str(parent_dir),
        features=builder.info.features,
        split_infos=builder.info.splits.values(),
        filename_template=f"{args.dataset_name}-{{SPLIT}}.tfrecord-{{SHARD_X_OF_Y}}",
        check_data=False,
    )

    features = load_features(version_dir)
    features = ensure_action_id_feature(features)
    save_features(args.output_dir, features)

    with open(version_dir / "dataset_info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    info["name"] = args.dataset_name
    with open(args.output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"Prepared rewritten metadata at: {args.output_dir}")
    print("Note: TFRecord shard rewriting still needs a dataset-export path compatible with your TFDS environment.")


if __name__ == "__main__":
    main()
