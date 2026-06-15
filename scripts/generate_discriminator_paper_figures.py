import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "action_id_discriminator_evaluation"
SUMMARY_CSV = EVAL_DIR / "summary_top1_4subsets.csv"
OUT_DIR = ROOT / "figure"

CM_OUT_PNG = OUT_DIR / "discriminator_confusion_4sets.png"
CM_OUT_PDF = OUT_DIR / "discriminator_confusion_4sets.pdf"
ERR_OUT_PNG = OUT_DIR / "discriminator_error_analysis.png"
ERR_OUT_PDF = OUT_DIR / "discriminator_error_analysis.pdf"
ERR_OK_OUT_PNG = OUT_DIR / "discriminator_error_analysis_ok.png"
ERR_OK_OUT_PDF = OUT_DIR / "discriminator_error_analysis_ok.pdf"

ORDER = [
    ("libero_object_action_id", "Object", "libero_object_action_id_train"),
    ("libero_goal_action_id", "Goal", "libero_goal_action_id_train"),
    ("libero_spatial_action_id", "Spatial", "libero_spatial_action_id_train"),
    ("libero_10_action_id", "LIBERO-10", "libero_10_action_id_train"),
]

ACTION_METADATA = [
    (0, "approach", "靠近", "#F7BA1E"),
    (1, "posture_adjustment", "姿态预调整", "#F9C74F"),
    (2, "retract", "复位", "#EC4899"),
    (3, "pull", "拉拽", "#FFA39E"),
    (4, "push", "推送", "#D4380D"),
    (5, "pick", "拿起", "#FFC069"),
    (6, "place", "放置", "#AD8B00"),
    (7, "pour", "倾倒", "#D3F261"),
    (8, "twist", "扭转", "#FAAD14"),
    (9, "slide", "滑动", "#1890FF"),
    (10, "other_move", "other_move", "#F8E45C"),
    (11, "align", "对齐", "#06B6D4"),
    (12, "lift", "上升", "#0EA5E9"),
    (13, "lower", "下降", "#3B82F6"),
    (14, "tilt", "抓手倾斜", "#86B817"),
    (15, "rotate", "旋拧", "#389E0D"),
    (16, "flip", "前后翻动", "#9254DE"),
    (17, "press", "下压", "#5CDBD3"),
]


def set_publication_style():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.1,
            "lines.linewidth": 2.0,
        }
    )


def load_summary():
    rows = {}
    with SUMMARY_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["dataset"]] = row
    return rows


def as_float(row, key):
    return float(row[key])


def load_fused_18x18_confusion() -> np.ndarray:
    missing = []
    mats = []
    for _, _, folder in ORDER:
        p = EVAL_DIR / folder / "confusion_matrix_raw_counts.npy"
        if not p.exists():
            missing.append(str(p))
            continue
        mats.append(np.load(p))

    if missing:
        msg = [
            "Missing raw 18x18 confusion matrices.",
            "Please rerun validation for the 4 subsets after enabling confusion-matrix data export,",
            "then regenerate this figure.",
            "Missing files:",
            *[f"- {m}" for m in missing],
        ]
        raise FileNotFoundError("\n".join(msg))

    fused = np.sum(np.stack(mats, axis=0), axis=0)
    if fused.shape != (18, 18):
        raise ValueError(f"Expected fused confusion shape (18,18), got {fused.shape}")
    return fused


def make_fused_confusion_figure():
    cm = load_fused_18x18_confusion()
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)
    cm_pct = cm_norm * 100.0

    # 🎯 修改1：调整比例，右侧更紧凑，混淆矩阵更大
    fig, (ax, ax_meta) = plt.subplots(
        1,
        2,
        figsize=(16.8, 9.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [5.5, 1.5]},
    )

    im = ax.imshow(cm_pct, cmap="YlGn", vmin=0.0, vmax=100.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success (%)", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    ticks = np.arange(18)
    tick_labels = [str(i + 1) for i in range(18)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=18)
    ax.set_yticklabels(tick_labels, fontsize=18)
    ax.set_xlabel("Predicted action ID", fontsize=19)
    ax.set_ylabel("Ground-truth action ID", fontsize=19)

    ax.set_xticks(np.arange(-0.5, 18, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 18, 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=0.8, alpha=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(18):
        for j in range(18):
            v = cm_pct[i, j]
            if i == j or v >= 3.0:
                color = "#1F2937" if v < 62 else "white"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=16, color=color)

    top1 = np.trace(cm) / np.sum(cm)
    ax.set_title(
        f"Fused 18×18 Confusion Matrix Across 4 LIBERO Suites  |  Top-1={100*top1:.2f}%",
        fontweight="bold",
        fontsize=22,
        pad=18,
    )

    # 🎯 修改2：右侧动作映射表 - 删除颜色列 + 整体左移 + 压缩宽度
    ax_meta.axis("off")
    ax_meta.set_title("Action ID Mapping", fontsize=26, fontweight="bold", pad=10)
    ax_meta.set_xlim(0, 0.3)  # 缩小宽度
    ax_meta.set_ylim(18.8, -0.8)

    # 标题左移紧凑
    ax_meta.text(0.00, 0.1, "Action ID", fontsize=22, fontweight="bold", va="bottom")
    ax_meta.text(0.20, 0.1, "Action", fontsize=22, fontweight="bold", va="bottom")

    for row_idx, (_, eng, zh, color) in enumerate(ACTION_METADATA):
        y = row_idx + 0.9
        # 文本整体左移，删除颜色方块
        ax_meta.text(0.05, y, str(row_idx + 1), fontsize=20, va="center")
        ax_meta.text(0.20, y, eng, fontsize=20, va="center")

    fig.savefig(CM_OUT_PNG, dpi=360, bbox_inches="tight")
    fig.savefig(CM_OUT_PDF, bbox_inches="tight")


def make_error_figure(summary):
    labels = [item[1] for item in ORDER]
    strict = []
    semantic = []
    temporal = []
    boundary_gain = []
    catastrophic = []

    for dataset, _, _ in ORDER:
        row = summary[dataset]
        s = 100 * as_float(row, "top1_acc")
        se = 100 * as_float(row, "stage_acc")
        te = 100 * as_float(row, "boundary_tolerant_acc")
        ca = 100 * as_float(row, "catastrophic_run_rate")

        strict.append(s)
        semantic.append(se)
        temporal.append(te)
        boundary_gain.append(te - s)
        catastrophic.append(ca)

    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.6), constrained_layout=True)

    ax = axes[0]
    w = 0.2
    gap = 0.04
    ax.bar(x - (w + gap), strict, width=w, color="#2563EB", edgecolor="#D1D5DB", linewidth=0.8, label="Strict")
    ax.bar(x, semantic, width=w, color="#059669", edgecolor="#D1D5DB", linewidth=0.8, label="Semantic")
    ax.bar(x + (w + gap), temporal, width=w, color="#F97316", edgecolor="#D1D5DB", linewidth=0.8, label="Temporal-tolerant")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)", fontsize=15)
    ax.set_title("(a) Accuracy Under Increasing Tolerance", fontweight="bold", fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.16, linewidth=0.8)
    ax.legend(frameon=False, loc="upper left", ncol=3)

    ax = axes[1]
    ax.bar(x, boundary_gain, width=0.22, color="#22C55E", edgecolor="#D1D5DB", linewidth=0.8, label="Temporal gain (pp)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Temporal gain over strict (pp)", fontsize=15)
    ax.set_title("(b) Impact of Errors on Routing Stability", fontweight="bold", fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.16, linewidth=0.8)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        catastrophic,
        color="#1D4ED8",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="#1D4ED8",
        markeredgewidth=1.8,
        linewidth=2.0,
        label="Catastrophic error rate",
    )
    ax2.set_ylabel("Catastrophic error rate (%)", fontsize=15)
    ax2.set_ylim(6, 30)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")

    fig.suptitle("Not All Routing Errors Are Equal", fontsize=18, fontweight="bold")

    fig.savefig(ERR_OUT_PNG, dpi=360, bbox_inches="tight")
    fig.savefig(ERR_OUT_PDF, bbox_inches="tight")
    fig.savefig(ERR_OK_OUT_PNG, dpi=360, bbox_inches="tight")
    fig.savefig(ERR_OK_OUT_PDF, bbox_inches="tight")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_publication_style()
    summary = load_summary()

    make_error_figure(summary)
    make_fused_confusion_figure()

    print("Figures saved.")


if __name__ == "__main__":
    main()