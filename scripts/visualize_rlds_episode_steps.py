import argparse
import math
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_episode(tfds_dir: Path, split: str, episode_index: int) -> Dict[str, Any]:
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError(
            "tensorflow_datasets is required to read RLDS datasets. "
            "Please install it in the current environment."
        ) from exc

    builder = tfds.builder_from_directory(str(tfds_dir))
    ds = builder.as_dataset(split=split)

    for idx, episode in enumerate(tfds.as_numpy(ds)):
        if idx == episode_index:
            return episode

    raise IndexError(f"Episode index {episode_index} is out of range for split '{split}'.")


def to_image(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    out = np.asarray(arr)
    if out.ndim != 3:
        return None
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def summarize_step(step: Dict[str, Any]) -> str:
    obs = step.get("observation", {})
    parts = []

    if "state" in obs:
        state = np.asarray(obs["state"]).reshape(-1)
        preview = ", ".join(f"{v:.3f}" for v in state[:4])
        parts.append(f"state[:4]=[{preview}]")

    if "action" in step:
        action = np.asarray(step["action"]).reshape(-1)
        preview = ", ".join(f"{v:.3f}" for v in action[:4])
        parts.append(f"action[:4]=[{preview}]")

    if "language_instruction" in step:
        lang = step["language_instruction"]
        if isinstance(lang, bytes):
            lang = lang.decode("utf-8", errors="ignore")
        if isinstance(lang, np.ndarray):
            lang = lang.item()
        if isinstance(lang, bytes):
            lang = lang.decode("utf-8", errors="ignore")
        parts.append(f"lang={lang}")

    if bool(step.get("is_first", False)):
        parts.append("is_first=True")
    if bool(step.get("is_last", False)):
        parts.append("is_last=True")
    if bool(step.get("is_terminal", False)):
        parts.append("is_terminal=True")

    return " | ".join(parts)


def save_step_panels(
    episode: Dict[str, Any],
    output_dir: Path,
    episode_index: int,
    max_steps: Optional[int],
) -> int:
    steps = list(episode["steps"])
    if max_steps is not None:
        steps = steps[:max_steps]

    output_dir.mkdir(parents=True, exist_ok=True)

    for step_idx, step in enumerate(steps):
        obs = step.get("observation", {})
        main_image = to_image(obs.get("image"))
        wrist_image = to_image(obs.get("wrist_image"))

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Episode {episode_index} | Step {step_idx}", fontsize=14)

        views = [
            (axes[0], main_image, "Main Image"),
            (axes[1], wrist_image, "Wrist Image"),
        ]
        for ax, image, title in views:
            ax.set_title(title, fontsize=11)
            ax.axis("off")
            if image is None:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=12)
            else:
                ax.imshow(image)

        footer = summarize_step(step)
        if footer:
            fig.text(0.02, 0.02, footer, fontsize=9, wrap=True)

        out_path = output_dir / f"episode_{episode_index:04d}_step_{step_idx:04d}.png"
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig.savefig(out_path, dpi=140)
        plt.close(fig)

    return len(steps)


def save_contact_sheet(
    episode: Dict[str, Any],
    output_dir: Path,
    episode_index: int,
    max_steps: Optional[int],
    columns: int,
) -> Optional[Path]:
    steps = list(episode["steps"])
    if max_steps is not None:
        steps = steps[:max_steps]
    if not steps:
        return None

    columns = max(1, columns)
    rows = math.ceil(len(steps) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 3.0))
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(rows, columns)

    fig.suptitle(f"Episode {episode_index} Contact Sheet", fontsize=16)

    for idx, step in enumerate(steps):
        r = idx // columns
        c = idx % columns
        ax = axes[r, c]
        obs = step.get("observation", {})
        main_image = to_image(obs.get("image"))
        wrist_image = to_image(obs.get("wrist_image"))

        ax.axis("off")
        ax.set_title(f"step {idx}", fontsize=9)
        if main_image is not None:
            ax.imshow(main_image)
            if wrist_image is not None:
                h, w = wrist_image.shape[:2]
                inset_h = max(1, h // 3)
                inset_w = max(1, w // 3)
                inset = wrist_image[:inset_h, :inset_w]
                ax.imshow(
                    inset,
                    extent=(0, main_image.shape[1] * 0.33, main_image.shape[0] * 0.33, 0),
                )
        elif wrist_image is not None:
            ax.imshow(wrist_image)
        else:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=12)

    total_axes = rows * columns
    for idx in range(len(steps), total_axes):
        r = idx // columns
        c = idx % columns
        axes[r, c].axis("off")

    out_path = output_dir / f"episode_{episode_index:04d}_contact_sheet.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize an RLDS episode step-by-step.")
    parser.add_argument("--tfds-dir", type=Path, required=True, help="Path to TFDS/RLDS dataset version directory.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load.")
    parser.add_argument("--episode-index", type=int, default=0, help="Zero-based episode index.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max number of steps to export.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "rlds_episode_viz",
        help="Directory to save per-step images.",
    )
    parser.add_argument(
        "--contact-columns",
        type=int,
        default=5,
        help="Number of columns in the contact sheet.",
    )
    args = parser.parse_args()

    episode = load_episode(args.tfds_dir, args.split, args.episode_index)
    output_dir = args.output_dir / f"episode_{args.episode_index:04d}"
    num_steps = save_step_panels(episode, output_dir, args.episode_index, args.max_steps)
    contact_sheet = save_contact_sheet(
        episode,
        output_dir,
        args.episode_index,
        args.max_steps,
        args.contact_columns,
    )

    print(f"Saved {num_steps} step images to: {output_dir}")
    if contact_sheet is not None:
        print(f"Saved contact sheet to: {contact_sheet}")


if __name__ == "__main__":
    main()
