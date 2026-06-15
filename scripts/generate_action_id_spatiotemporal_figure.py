import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TFDS_DIR = ROOT / "libero_action_id" / "0.3.2"
DEFAULT_OUT = ROOT / "figure" / "action_id_spatiotemporal_structure.png"

GROUPS = [
    {"name": "Independent Motion", "ids": set(range(0, 3)), "color": "#1F77B4", "y": 0},
    {"name": "Interaction", "ids": set(range(3, 11)), "color": "#F28E2B", "y": 1},
    {"name": "Fine Adjustment", "ids": None, "color": "#59A14F", "y": 2},
]


def set_publication_style():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.0,
        }
    )


def group_name(action_id: int) -> str:
    aid = int(action_id)
    for group in GROUPS[:2]:
        if aid in group["ids"]:
            return group["name"]
    return GROUPS[2]["name"]


def load_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"x", "y", "z", "action_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    if "episode_id" not in df.columns:
        df["episode_id"] = 0

    if "t_norm" not in df.columns:
        if "timestep" in df.columns:
            time_col = "timestep"
        elif "step" in df.columns:
            time_col = "step"
        elif "t" in df.columns:
            time_col = "t"
        else:
            df["timestep"] = df.groupby("episode_id").cumcount()
            time_col = "timestep"

        df["t_norm"] = 0.0
        for _, idx in df.groupby("episode_id").groups.items():
            vals = df.loc[idx, time_col].to_numpy(dtype=float)
            if len(vals) <= 1 or vals.max() == vals.min():
                df.loc[idx, "t_norm"] = 0.0
            else:
                df.loc[idx, "t_norm"] = 2.0 * (vals - vals.min()) / (vals.max() - vals.min()) - 1.0

    return standardize_df(df)


def load_from_tfds(tfds_dir: Path, max_episodes: Optional[int] = None) -> pd.DataFrame:
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError(
            "tensorflow_datasets is required for --tfds-dir mode. "
            "Install it or export a CSV and use --csv instead."
        ) from exc

    builder = tfds.builder_from_directory(str(tfds_dir))
    ds = builder.as_dataset(split="train")

    rows = []
    for ep_idx, episode in enumerate(tfds.as_numpy(ds)):
        if max_episodes is not None and ep_idx >= max_episodes:
            break

        steps = list(episode["steps"])
        n_steps = len(steps)
        if n_steps == 0:
            continue

        for step_idx, step in enumerate(steps):
            obs = step["observation"]
            state = np.asarray(obs["state"], dtype=float)
            action_id = int(np.asarray(obs["action_id"]))
            rows.append(
                {
                    "episode_id": ep_idx,
                    "timestep": step_idx,
                    "t_norm": 0.0 if n_steps <= 1 else 2.0 * step_idx / (n_steps - 1) - 1.0,
                    "x": state[0],
                    "y": state[1],
                    "z": state[2],
                    "action_id": action_id,
                }
            )

    if not rows:
        raise RuntimeError(f"No trajectory rows loaded from {tfds_dir}")
    return standardize_df(pd.DataFrame(rows))


def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["x", "y", "z", "action_id", "t_norm"])
    out["action_id"] = out["action_id"].astype(int)
    out["group"] = out["action_id"].map(group_name)
    return out


def downsample_group(df: pd.DataFrame, max_points_per_group: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for group in GROUPS:
        sub = df[df["group"] == group["name"]]
        if len(sub) > max_points_per_group:
            take = rng.choice(sub.index.to_numpy(), size=max_points_per_group, replace=False)
            sub = sub.loc[take]
        parts.append(sub)
    return pd.concat(parts, axis=0).reset_index(drop=True)


def smooth_histogram(values: np.ndarray, bins: np.ndarray, sigma_bins: float = 2.2):
    hist, edges = np.histogram(values, bins=bins, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    radius = max(1, int(4 * sigma_bins))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(hist.astype(float), kernel, mode="same")
    if smooth.max() > 0:
        smooth = smooth / smooth.max()
    return centers, smooth


def plot_figure(df: pd.DataFrame, out_path: Path, max_points_per_group: int, seed: int):
    set_publication_style()
    sampled = downsample_group(df, max_points_per_group=max_points_per_group, seed=seed)

    fig = plt.figure(figsize=(7.2, 10.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.18, 1.0])
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_time = fig.add_subplot(gs[1, 0])

    for group in GROUPS:
        sub_full = df[df["group"] == group["name"]]
        sub = sampled[sampled["group"] == group["name"]]
        label = f'{group["name"]} (n={len(sub_full)})'
        ax3d.scatter(
            sub["x"],
            sub["y"],
            sub["z"],
            s=5,
            color=group["color"],
            alpha=0.34,
            edgecolors="none",
            label=label,
        )

    ax3d.set_xlabel("X Position")
    ax3d.set_ylabel("Y Position")
    ax3d.set_zlabel("Z Position")
    ax3d.view_init(elev=24, azim=-58)
    ax3d.legend(frameon=True, framealpha=0.92, loc="upper right", borderpad=0.6)
    ax3d.text2D(
        0.18,
        -0.08,
        "(a) Spatial distribution of action groups",
        transform=ax3d.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )

    bins = np.linspace(-1.0, 1.0, 120)
    rng = np.random.default_rng(seed)
    for group in GROUPS:
        sub_full = df[df["group"] == group["name"]]
        y0 = float(group["y"])
        if sub_full.empty:
            continue

        centers, density = smooth_histogram(sub_full["t_norm"].to_numpy(dtype=float), bins)
        density_height = 0.42 * density
        ax_time.fill_between(centers, y0, y0 + density_height, color=group["color"], alpha=0.24, linewidth=0)
        ax_time.plot(centers, y0 + density_height, color=group["color"], linewidth=1.7)

        scatter_sub = sub_full
        if len(scatter_sub) > max_points_per_group:
            take = rng.choice(scatter_sub.index.to_numpy(), size=max_points_per_group, replace=False)
            scatter_sub = scatter_sub.loc[take]
        jitter = rng.normal(0.0, 0.055, size=len(scatter_sub))
        ax_time.scatter(
            scatter_sub["t_norm"],
            y0 + jitter,
            s=5,
            color=group["color"],
            alpha=0.22,
            edgecolors="none",
        )

    ax_time.set_xlabel("Normalized episode progress")
    ax_time.set_ylabel("Action group")
    ax_time.set_xlim(-1.02, 1.02)
    ax_time.set_ylim(-0.55, len(GROUPS) - 0.25)
    ax_time.set_yticks([g["y"] for g in GROUPS])
    ax_time.set_yticklabels([g["name"] for g in GROUPS])
    ax_time.grid(axis="x", linestyle="--", alpha=0.25, linewidth=0.8)
    ax_time.spines["top"].set_visible(False)
    ax_time.spines["right"].set_visible(False)
    ax_time.text(
        0.5,
        -0.22,
        "(b) Temporal distribution of action groups",
        transform=ax_time.transAxes,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )

    fig.suptitle("Spatio-temporal Structure of Action Groups on LIBERO", fontsize=16, fontweight="bold", y=1.015)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=360, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Rows loaded: {len(df)}")
    print(f"Saved: {out_path}")
    print(f"Saved: {out_path.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a paper-style figure showing spatio-temporal structure of LIBERO action groups."
    )
    parser.add_argument("--csv", type=Path, default=None, help="CSV with x,y,z,action_id and optional episode_id/timestep/t_norm.")
    parser.add_argument("--tfds-dir", type=Path, default=DEFAULT_TFDS_DIR, help="Local TFDS/RLDS directory for libero_action_id.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output PNG path. A PDF with the same stem is also saved.")
    parser.add_argument("--max-episodes", type=int, default=250, help="Max episodes to load in TFDS mode.")
    parser.add_argument("--max-points-per-group", type=int, default=12000, help="Max plotted points per action group.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.csv is not None:
        df = load_from_csv(args.csv)
    else:
        df = load_from_tfds(args.tfds_dir, max_episodes=args.max_episodes)

    plot_figure(df, args.out, max_points_per_group=args.max_points_per_group, seed=args.seed)


if __name__ == "__main__":
    main()
