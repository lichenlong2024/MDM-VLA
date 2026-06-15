import csv
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "action_id_discriminator_evaluation"
SUMMARY_CSV = EVAL_DIR / "summary_top1_4subsets.csv"
OUTPUT_DIR = ROOT / "figure"
OUTPUT_PNG = OUTPUT_DIR / "discriminator_summary.png"
OUTPUT_PDF = OUTPUT_DIR / "discriminator_summary.pdf"
REPRESENTATIVE_SUITE = "libero_spatial_action_id"
REPRESENTATIVE_CM = EVAL_DIR / "libero_spatial_action_id_train" / "confusion_matrix_normalized_True.png"

DISPLAY_NAMES = {
    "libero_object_action_id": "Object",
    "libero_goal_action_id": "Goal",
    "libero_spatial_action_id": "Spatial",
    "libero_10_action_id": "LIBERO-10",
}

ORDER = [
    "libero_object_action_id",
    "libero_goal_action_id",
    "libero_spatial_action_id",
    "libero_10_action_id",
]

COLORS = {
    "top1": "#2563EB",
    "stage": "#059669",
    "boundary": "#DC2626",
    "within": "#60A5FA",
    "cross": "#F59E0B",
    "run": "#7C3AED",
    "catastrophic": "#EF4444",
    "grid": "#D1D5DB",
    "text": "#111827",
    "muted": "#6B7280",
}


def load_summary():
    rows = {}
    with SUMMARY_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row["dataset"]
            rows[dataset] = {k: float(v) if k not in {"dataset", "split", "output_dir"} else v for k, v in row.items()}
    return rows


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_panel_label(ax, label):
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=COLORS["text"],
        va="top",
    )


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.tick_params(colors=COLORS["text"], labelsize=10)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35, color=COLORS["grid"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()

    suites = []
    top1 = []
    stage = []
    boundary = []
    cross = []
    within = []
    mean_run = []
    catastrophic = []

    for dataset in ORDER:
        row = summary[dataset]
        out_dir = Path(row["output_dir"])
        _ = load_json(EVAL_DIR / out_dir.name / "mechanism_metrics.json")
        _ = load_json(EVAL_DIR / out_dir.name / "temporal_error_report.json")

        suites.append(DISPLAY_NAMES[dataset])
        top1.append(row["top1_acc"] * 100)
        stage.append(row["stage_acc"] * 100)
        boundary.append(row["boundary_tolerant_acc"] * 100)
        cross_ratio = row["cross_stage_err_ratio"] * 100
        cross.append(cross_ratio)
        within.append(100 - cross_ratio)
        mean_run.append(row["mean_error_run_len"])
        catastrophic.append(row["catastrophic_run_rate"] * 100)

    y = np.arange(len(suites))

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(14, 9.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.08, 1], height_ratios=[1, 1])

    # (a) Representative confusion matrix image
    ax_cm = fig.add_subplot(gs[0, 0])
    img = mpimg.imread(REPRESENTATIVE_CM)
    ax_cm.imshow(img)
    ax_cm.axis("off")
    ax_cm.set_title("Representative Fine-Grained Confusion Matrix (Spatial)", color=COLORS["text"])
    add_panel_label(ax_cm, "(a)")

    # (b) Accuracy ladder plot
    ax_acc = fig.add_subplot(gs[0, 1])
    style_axis(ax_acc)
    add_panel_label(ax_acc, "(b)")
    ax_acc.set_title("Accuracy Improves with Semantic and Temporal Relaxation", color=COLORS["text"])
    for i, suite in enumerate(suites):
        ax_acc.plot([top1[i], stage[i], boundary[i]], [i, i, i], color="#CBD5E1", linewidth=2.0, zorder=1)
        ax_acc.scatter(top1[i], i, s=58, color=COLORS["top1"], zorder=3)
        ax_acc.scatter(stage[i], i, s=58, color=COLORS["stage"], zorder=3)
        ax_acc.scatter(boundary[i], i, s=58, color=COLORS["boundary"], zorder=3)
        ax_acc.text(boundary[i] + 0.55, i, f"{boundary[i]:.1f}", va="center", fontsize=9.5, color=COLORS["muted"])
    ax_acc.set_yticks(y)
    ax_acc.set_yticklabels(suites)
    ax_acc.set_xlim(78, 96.5)
    ax_acc.set_xlabel("Accuracy (%)")
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["top1"], markersize=8, label="Top-1"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["stage"], markersize=8, label="Stage"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["boundary"], markersize=8, label="Boundary-tolerant"),
    ]
    ax_acc.legend(handles=legend_handles, frameon=False, ncol=3, loc="lower right")

    # (c) Error composition stacked horizontal bars
    ax_err = fig.add_subplot(gs[1, 0])
    style_axis(ax_err)
    add_panel_label(ax_err, "(c)")
    ax_err.set_title("Most Errors Stay Within the Correct Semantic Stage", color=COLORS["text"])
    ax_err.barh(y, within, color=COLORS["within"], height=0.55, label="Within-stage")
    ax_err.barh(y, cross, left=within, color=COLORS["cross"], height=0.55, label="Cross-stage")
    ax_err.set_yticks(y)
    ax_err.set_yticklabels(suites)
    ax_err.set_xlim(0, 100)
    ax_err.set_xlabel("Fraction of misclassified frames (%)")
    for i in range(len(suites)):
        ax_err.text(within[i] / 2, i, f"{within[i]:.1f}", ha="center", va="center", fontsize=9, color=COLORS["text"])
        ax_err.text(within[i] + cross[i] / 2, i, f"{cross[i]:.1f}", ha="center", va="center", fontsize=9, color=COLORS["text"])
    ax_err.legend(frameon=False, loc="lower right")

    # (d) Temporal severity scatter with twin axis
    ax_tmp = fig.add_subplot(gs[1, 1])
    style_axis(ax_tmp)
    add_panel_label(ax_tmp, "(d)")
    ax_tmp.set_title("Temporal Failures Are Mostly Short, Not Persistent", color=COLORS["text"])
    ax_tmp.plot(y, mean_run, color=COLORS["run"], marker="o", linewidth=2.2, markersize=6.5, label="Mean error run")
    ax_tmp.set_xticks(y)
    ax_tmp.set_xticklabels(suites)
    ax_tmp.set_ylabel("Mean error run length (frames)", color=COLORS["run"])
    ax_tmp.tick_params(axis="y", labelcolor=COLORS["run"])
    ax_tmp.set_ylim(3.5, 6.2)

    ax_tmp2 = ax_tmp.twinx()
    ax_tmp2.spines["top"].set_visible(False)
    ax_tmp2.spines["left"].set_visible(False)
    ax_tmp2.spines["right"].set_color(COLORS["grid"])
    ax_tmp2.tick_params(colors=COLORS["catastrophic"], labelsize=10)
    ax_tmp2.plot(y, catastrophic, color=COLORS["catastrophic"], marker="s", linewidth=2.2, markersize=6, label="Catastrophic run rate")
    ax_tmp2.set_ylabel("Catastrophic run rate (%)", color=COLORS["catastrophic"])
    ax_tmp2.set_ylim(12, 27)

    combo_handles = [
        Line2D([0], [0], color=COLORS["run"], marker="o", linewidth=2.2, markersize=6.5, label="Mean error run"),
        Line2D([0], [0], color=COLORS["catastrophic"], marker="s", linewidth=2.2, markersize=6, label="Catastrophic run rate"),
    ]
    ax_tmp.legend(handles=combo_handles, frameon=False, loc="upper left")

    fig.suptitle(
        "Why the Discriminator Is a Useful Routing Prior",
        fontsize=16,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
