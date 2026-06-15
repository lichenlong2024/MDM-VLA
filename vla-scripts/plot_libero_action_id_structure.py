"""Plot single-task action-ID structure for LIBERO RLDS datasets.

This script scans LIBERO-10 action-ID datasets, selects the task whose
stage-labeled action chunks show the clearest separation, and creates a
publication-oriented diagnostic figure.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds


STAGE_NAMES = {
    0: "S1",
    1: "S2",
    2: "S3",
}

STAGE_COLORS = {
    0: "#2F80ED",
    1: "#F2994A",
    2: "#27AE60",
}


def action_id_to_stage(action_id):
    action_id = int(action_id)
    if action_id <= 2:
        return 0
    if action_id <= 10:
        return 1
    return 2


def iter_episodes(dataset_dir):
    builder = tfds.builder_from_directory(str(dataset_dir))
    ds = builder.as_dataset(split="train")
    for ep in ds:
        actions, states, action_ids, instructions = [], [], [], []
        for step in tfds.as_numpy(ep["steps"]):
            actions.append(step["action"].astype(np.float32))
            states.append(step["observation"]["state"].astype(np.float32))
            action_ids.append(int(step["observation"]["action_id"]))
            instr = step.get("language_instruction", b"")
            if isinstance(instr, bytes):
                instr = instr.decode("utf-8", errors="ignore")
            instructions.append(str(instr))
        if actions:
            yield {
                "actions": np.stack(actions),
                "states": np.stack(states),
                "action_ids": np.array(action_ids, dtype=np.int64),
                "instruction": instructions[0] if instructions else "",
            }


def load_task(dataset_dir, max_episodes=None):
    episodes = []
    for idx, ep in enumerate(iter_episodes(dataset_dir)):
        episodes.append(ep)
        if max_episodes is not None and idx + 1 >= max_episodes:
            break
    if not episodes:
        raise RuntimeError(f"No episodes found in {dataset_dir}")
    return episodes


def collect_points(episodes, chunk_len=8, stride=2):
    chunks, actions, states, stages, action_ids, progress, ep_ids = [], [], [], [], [], [], []
    for ep_idx, ep in enumerate(episodes):
        a = ep["actions"]
        s = ep["states"]
        ids = ep["action_ids"]
        n = len(ids)
        if n < chunk_len:
            continue
        denom = max(n - 1, 1)
        for t in range(0, n - chunk_len + 1, stride):
            aid = int(ids[t])
            stage = action_id_to_stage(aid)
            chunk = a[t : t + chunk_len].reshape(-1)
            chunks.append(chunk)
            actions.append(a[t])
            states.append(s[t])
            stages.append(stage)
            action_ids.append(aid)
            progress.append(t / denom)
            ep_ids.append(ep_idx)
    return {
        "chunks": np.asarray(chunks, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "states": np.asarray(states, dtype=np.float32),
        "stages": np.asarray(stages, dtype=np.int64),
        "action_ids": np.asarray(action_ids, dtype=np.int64),
        "progress": np.asarray(progress, dtype=np.float32),
        "ep_ids": np.asarray(ep_ids, dtype=np.int64),
    }


def pca_2d(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = x / np.maximum(std, 1e-6)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    return (x @ vh[:2].T).astype(np.float32)


def separation_score(points):
    x = points["chunks"]
    y = points["stages"]
    valid_stages = [s for s in sorted(set(y.tolist())) if np.sum(y == s) >= 20]
    if len(valid_stages) < 2:
        return -1.0
    # Use normalized chunk vectors for a simple Fisher-style separability score.
    x = (x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), 1e-6)
    global_mean = x.mean(axis=0)
    between, within = 0.0, 0.0
    total = len(x)
    for s in valid_stages:
        xs = x[y == s]
        mean = xs.mean(axis=0)
        between += len(xs) * np.sum((mean - global_mean) ** 2)
        within += np.sum((xs - mean) ** 2)
    return float((between / max(len(valid_stages) - 1, 1)) / (within / max(total - len(valid_stages), 1) + 1e-8))


def summarize_task(task_name, episodes, points):
    counts = {STAGE_NAMES[s]: int(np.sum(points["stages"] == s)) for s in range(3)}
    action_counts = {
        str(i): int(np.sum(points["action_ids"] == i))
        for i in sorted(set(points["action_ids"].tolist()))
    }
    instruction = episodes[0]["instruction"] if episodes else ""
    return {
        "task": task_name,
        "instruction": instruction,
        "num_episodes": len(episodes),
        "num_chunks": int(len(points["stages"])),
        "stage_counts": counts,
        "action_id_counts": action_counts,
        "separation_score": separation_score(points),
    }


def scan_libero10(data_root, chunk_len, stride, max_episodes=None, min_stage_count=40):
    candidates = []
    for task_dir in sorted(Path(data_root).glob("libero_10_*_action_id")):
        # Skip the merged suite when looking for a single clear task.
        if task_dir.name == "libero_10_rlds_action_id":
            continue
        dataset_dir = task_dir / "shuju"
        if not dataset_dir.exists():
            continue
        episodes = load_task(dataset_dir, max_episodes=max_episodes)
        points = collect_points(episodes, chunk_len=chunk_len, stride=stride)
        if len(points["stages"]) == 0:
            continue
        summary = summarize_task(task_dir.name, episodes, points)
        raw_counts = list(summary["stage_counts"].values())
        stage_presence = sum(v >= min_stage_count for v in raw_counts)
        min_count = min(raw_counts)
        candidates.append((stage_presence, min_count, summary["separation_score"], task_dir.name, episodes, points, summary))
    if not candidates:
        raise RuntimeError(f"No LIBERO-10 action-ID datasets found under {data_root}")
    # Prefer tasks with more observed stages, then stronger action-space separation.
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates


def downsample_by_stage(points, max_per_stage=900, seed=7):
    rng = np.random.default_rng(seed)
    keep = []
    for s in range(3):
        idx = np.flatnonzero(points["stages"] == s)
        if len(idx) > max_per_stage:
            idx = rng.choice(idx, size=max_per_stage, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return {k: v[keep] for k, v in points.items()}


def plot_structure(points, summary, out_png, out_pdf=None):
    plot_points = downsample_by_stage(points)
    stages = plot_points["stages"]
    progress = plot_points["progress"]
    actions = plot_points["actions"]
    action_scale = np.linalg.norm(actions[:, :6], axis=1)

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 7.6,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.6,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "figure.dpi": 160,
        "savefig.dpi": 600,
    })

    fig = plt.figure(figsize=(3.65, 1.82))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.31)

    ax1 = fig.add_subplot(gs[0, 0])
    jitter_map = {0: 0.0, 1: 1.0, 2: 2.0}
    rng = np.random.default_rng(12)
    for s in range(3):
        idx = stages == s
        if np.any(idx):
            y = np.full(np.sum(idx), jitter_map[s]) + rng.normal(0, 0.035, np.sum(idx))
            ax1.scatter(
                progress[idx],
                y,
                s=3.7,
                alpha=0.26,
                c=STAGE_COLORS[s],
                edgecolors="none",
            )
            bins = np.linspace(0, 1, 26)
            hist, edges = np.histogram(progress[idx], bins=bins)
            if hist.max() > 0:
                hist = hist / hist.max() * 0.30
                centers = (edges[:-1] + edges[1:]) / 2
                ax1.plot(centers, jitter_map[s] + hist + 0.08, color=STAGE_COLORS[s], linewidth=0.85)
    ax1.set_title("(a) Temporal structure", fontweight="bold")
    ax1.set_xlabel("Normalized episode progress")
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(["S1", "S2", "S3"])
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.35, 2.55)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, axis="x", color="#E5E7EB", linewidth=0.6)

    ax2 = fig.add_subplot(gs[0, 1])
    box_data, labels, colors = [], [], []
    for s in range(3):
        vals = action_scale[stages == s]
        if len(vals) == 0:
            continue
        # Clip only the plotting tail to avoid one rare outlier dominating the axis.
        hi = np.percentile(vals, 97.5)
        vals = np.clip(vals, None, hi)
        box_data.append(vals)
        labels.append(["S1", "S2", "S3"][s])
        colors.append(STAGE_COLORS[s])
    bp = ax2.boxplot(
        box_data,
        vert=False,
        patch_artist=True,
        widths=0.42,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.05},
        boxprops={"linewidth": 0.75, "color": "#4B5563"},
        whiskerprops={"linewidth": 0.75, "color": "#6B7280"},
        capprops={"linewidth": 0.75, "color": "#6B7280"},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.58)
    ax2.set_title("(b) Action scale", fontweight="bold")
    ax2.set_yticks(np.arange(1, len(labels) + 1))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel(r"$\|a_{xyz,rpy}\|_2$")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, axis="x", color="#E5E7EB", linewidth=0.6)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=STAGE_COLORS[s],
                   markeredgecolor="none", markersize=4.5, label=STAGE_NAMES[s])
        for s in range(3)
    ]
    fig.legend(legend_handles, [STAGE_NAMES[s] for s in range(3)],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.subplots_adjust(bottom=0.30, top=0.86, left=0.12, right=0.985)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"D:\VLAproject\liberorelatedata")
    parser.add_argument("--output", default="figure/action_id_single_task_structure.png")
    parser.add_argument("--output-pdf", default="figure/action_id_single_task_structure.pdf")
    parser.add_argument("--summary", default="figure/action_id_single_task_structure_summary.json")
    parser.add_argument("--task", default=None, help="Optional dataset name, e.g. libero_10_1_action_id.")
    parser.add_argument("--chunk-len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--min-stage-count", type=int, default=40)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.task:
        task_dir = data_root / args.task / "shuju"
        episodes = load_task(task_dir, max_episodes=args.max_episodes)
        points = collect_points(episodes, chunk_len=args.chunk_len, stride=args.stride)
        summary = summarize_task(args.task, episodes, points)
    else:
        candidates = scan_libero10(
            data_root,
            args.chunk_len,
            args.stride,
            args.max_episodes,
            min_stage_count=args.min_stage_count,
        )
        _, _, _, _, episodes, points, summary = candidates[0]
        summary["candidate_ranking"] = [
            {
                "task": c[3],
                "stage_presence": int(c[0]),
                "min_stage_count": int(c[1]),
                "separation_score": float(c[2]),
                "stage_counts": c[6]["stage_counts"],
                "instruction": c[6]["instruction"],
            }
            for c in candidates
        ]

    plot_structure(points, summary, args.output, args.output_pdf)
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
