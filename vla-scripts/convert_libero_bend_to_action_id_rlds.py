import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable


def build_action_id_rules():
    # Applies to all episodes:
    # 0-34   -> 0
    # 35-74  -> 5
    # 75-end -> 15
    def label_for_step(step_idx: int) -> int:
        if step_idx < 35:
            return 0
        if step_idx < 75:
            return 5
        return 15

    return label_for_step


def require_tf():
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise RuntimeError(
            "This script must be run in an environment that has both "
            "`tensorflow` and `tensorflow_datasets` installed."
        ) from exc
    return tf, tfds


def make_builder(source_version_dir: Path, output_root: Path, dataset_name: str):
    tf, tfds = require_tf()
    source_builder = tfds.builder_from_directory(str(source_version_dir))
    label_for_step = build_action_id_rules()
    from tensorflow_datasets.core import dataset_metadata

    class LiberoBendActionId(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {"1.0.0": "Converted from libero_bend by adding per-step action_id labels."}

        @classmethod
        def get_metadata(cls):
            # Running as a standalone script means TFDS cannot resolve package metadata
            # from importlib resources. Returning an empty metadata object avoids that path.
            return dataset_metadata.DatasetMetadata(
                description="Converted from libero_bend by adding per-step action_id labels.",
                citation="",
                tags=[],
            )

        def _info(self):
            return self.dataset_info_from_configs(
                features=tfds.features.FeaturesDict(
                    {
                        "episode_metadata": tfds.features.FeaturesDict(
                            {
                                "file_path": tfds.features.Text(),
                            }
                        ),
                        "steps": tfds.features.Dataset(
                            tfds.features.FeaturesDict(
                                {
                                    "action": tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                                    "observation": tfds.features.FeaturesDict(
                                        {
                                            "image": tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8, encoding_format="jpeg"),
                                            "wrist_image": tfds.features.Image(
                                                shape=(256, 256, 3), dtype=tf.uint8, encoding_format="jpeg"
                                            ),
                                            "state": tfds.features.Tensor(shape=(8,), dtype=tf.float32),
                                            "joint_state": tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                                            "action_id": tfds.features.Tensor(shape=(), dtype=tf.int64),
                                        }
                                    ),
                                    "is_first": tf.bool,
                                    "is_last": tf.bool,
                                    "is_terminal": tf.bool,
                                    "language_instruction": tfds.features.Text(),
                                    "reward": tf.float32,
                                    "discount": tf.float32,
                                }
                            )
                        ),
                    }
                )
            )

        def _split_generators(self, dl_manager):
            return {
                "train": self._generate_examples("train"),
            }

        def _generate_examples(self, split: str) -> Iterable:
            ds = source_builder.as_dataset(split=split)
            for ep_idx, episode in enumerate(tfds.as_numpy(ds)):
                steps = []
                for step_idx, step in enumerate(episode["steps"]):
                    obs = step["observation"]
                    steps.append(
                        {
                            "action": step["action"],
                            "observation": {
                                "image": obs["image"],
                                "wrist_image": obs["wrist_image"],
                                "state": obs["state"],
                                "joint_state": obs["joint_state"],
                                "action_id": label_for_step(step_idx),
                            },
                            "is_first": step["is_first"],
                            "is_last": step["is_last"],
                            "is_terminal": step["is_terminal"],
                            "language_instruction": step["language_instruction"],
                            "reward": step["reward"],
                            "discount": step["discount"],
                        }
                    )

                yield ep_idx, {
                    "episode_metadata": {
                        "file_path": episode["episode_metadata"]["file_path"],
                    },
                    "steps": steps,
                }

    LiberoBendActionId.__name__ = dataset_name
    return LiberoBendActionId(data_dir=str(output_root), config=None), tf, tfds


def main():
    parser = argparse.ArgumentParser(description="Convert libero_bend RLDS into libero_bend_action_id RLDS.")
    parser.add_argument(
        "--input-version-dir",
        type=Path,
        required=True,
        help="Path to the source TFDS/RLDS dataset directory, e.g., /path/to/tensorflow_datasets/libero_bend/1.0.0",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Parent directory where the new dataset folder will be written.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="libero_bend_action_id",
        help="Name of the converted dataset folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, remove existing output dataset directory before writing.",
    )
    args = parser.parse_args()

    if not args.input_version_dir.exists():
        raise FileNotFoundError(f"Input dataset version directory does not exist: {args.input_version_dir}")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    target_dataset_dir = output_root / args.dataset_name
    if args.overwrite and target_dataset_dir.exists():
        shutil.rmtree(target_dataset_dir)

    builder, tf, _tfds = make_builder(args.input_version_dir, output_root, args.dataset_name)
    builder.download_and_prepare(
        file_format="tfrecord",
    )

    built_path = output_root / args.dataset_name / str(builder.VERSION)
    print(f"Done. Converted dataset written under: {built_path}")
    print("Applied global action_id rules:")
    print("  steps 0-34   -> 0")
    print("  steps 35-74  -> 5")
    print("  steps 75-end -> 15")


if __name__ == "__main__":
    main()
