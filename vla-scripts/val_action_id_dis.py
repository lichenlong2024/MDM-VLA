"""
楠岃瘉Action ID鍒ゅ埆鍣ㄧ殑瀹屾暣鑴氭湰锛堢函鎵嬪姩璁＄畻娣锋穯鐭╅樀锛屾棤sklearn渚濊禆锛?
"""
import argparse
from huggingface_hub import HfApi, snapshot_download
import json
import os
import time
from collections import deque, defaultdict
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import numpy as np
from prismatic.models import load, load_vla
from prismatic.models.action_id_discriminator import ActionIDDiscriminator, ActionIDLoss
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM
from prismatic.models.action_heads import L1RegressionActionHead, StageMoEActionHead
from prismatic.models.projectors import ProprioProjector
from prismatic.models.backbones.vision.dinosiglip_vit import DinoSigLIPViTBackbone
from transformers import AutoModelForVision2Seq, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
import wandb
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask
)
from transformers import AutoModelForVision2Seq, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# 浠呬繚鐣欏熀纭€鍙鍖栦緷璧栵紙鏃爏klearn锛?
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 璁剧疆涓枃瀛椾綋锛堝彲閫夛級
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 璁剧疆鐜鍙橀噺
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"
os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "true"
from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map
)

# ==============================================================================
# 1. 閰嶇疆鍙傛暟 (Configuration)
# ==============================================================================
CONFIG = {
    # 璺緞 (Paths)
    "config_path": "<PATH_TO_VLA_CONFIG>",
    "vlm_path": "<PATH_TO_PRETRAINED_VLM>",
    "data_root_dir": "<PATH_TO_DATA_ROOT>",
    "dataset_name": "libero_bread_action_id",
    "checkpoint_path": "<PATH_TO_DISCRIMINATOR_CHECKPOINT>",
    "output_dir": "<PATH_TO_EVAL_OUTPUT_DIR>",

    # 妯″瀷鍙傛暟 (Model Parameters)
    "num_images_in_input": 2,
    "ACTION_DIM": 8,
    "proprio_dim": 8,
    "num_action_ids": 18,
    "vision_backbone_id": "dinosiglip-vit-so-224px",
    "image_resize_strategy": "resize-naive",
    "image_size": 224,
    "NUM_PATCHES": 196,

    # 楠岃瘉鍙傛暟 (Validation Parameters)
    "batch_size": 16,
    "val_time_limit": None,
    "use_vla_model": True,
    "use_moe_action_head": True,

    # 娣锋穯鐭╅樀閰嶇疆
    "cm_figsize": (16, 14),
    "cm_fontsize": 10,
    "cm_normalize": True,
    "cm_save_format": "png",
    "cm_dpi": 300,

    # 鏃跺簭閿欒鍒嗘瀽閰嶇疆
    "enable_temporal_error_analysis": True,
    "temporal_boundary_tolerance": 2,
    "catastrophic_error_run_threshold": 8,
    "segment_failure_ratio_threshold": 0.6,
    "shuffle_buffer_size": 1,

    # Stage 瀹氫箟锛堜笌 StageMoEActionHead 榛樿涓€鑷达級
    "stage_definitions": {
        0: [0, 1, 2],
        1: [3, 4, 5,6, 7, 8, 9, 10],
        2: [11,12, 13, 14, 15, 16, 17],
    },
}

# ==============================================================================
# 2. 绾墜鍔ㄥ疄鐜版贩娣嗙煩闃?鍒嗙被鎶ュ憡锛堟棤sklearn渚濊禆锛?
# ==============================================================================

def compute_confusion_matrix(y_true, y_pred, num_classes):
    """
    鎵嬪姩璁＄畻娣锋穯鐭╅樀
    Args:
        y_true: 鐪熷疄鏍囩鏁扮粍 (np.array)
        y_pred: 棰勬祴鏍囩鏁扮粍 (np.array)
        num_classes: 绫诲埆鎬绘暟
    Returns:
        cm: 娣锋穯鐭╅樀 (num_classes x num_classes)
    """
    # 鍒濆鍖栨贩娣嗙煩闃典负鍏?
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    # 閬嶅巻姣忎釜鏍锋湰锛岀粺璁＄湡瀹炴爣绛惧拰棰勬祴鏍囩鐨勭粍鍚?
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:  # 闃叉鏍囩瓒婄晫
            cm[t][p] += 1

    return cm

def normalize_confusion_matrix(cm):
    """鎵嬪姩褰掍竴鍖栨贩娣嗙煩闃碉紙鎸夎褰掍竴鍖栵級"""
    # 璁＄畻姣忚鐨勬€诲拰
    row_sums = cm.sum(axis=1, keepdims=True)
    # 閬垮厤闄や互0锛堝鐞嗘棤鏍锋湰鐨勭被鍒級
    row_sums[row_sums == 0] = 1
    # 褰掍竴鍖?
    cm_normalized = cm.astype(np.float32) / row_sums
    return np.round(cm_normalized, 4)

def compute_classification_metrics(y_true, y_pred, num_classes):
    """
    鎵嬪姩璁＄畻姣忕被鐨勭簿纭巼銆佸彫鍥炵巼銆丗1鍒嗘暟
    Returns:
        metrics: 瀛楀吀锛屽寘鍚瘡绫荤殑precision/recall/f1锛屼互鍙婂畯骞冲潎/寰钩鍧?
    """
    metrics = {
        "per_class": {},
        "macro_avg": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "micro_avg": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "support": {}  # 姣忕被鏍锋湰鏁?
    }

    # 璁＄畻姣忕被鐨凾P/FP/FN/TN
    tp = np.zeros(num_classes)  # 鐪熸渚?
    fp = np.zeros(num_classes)  # 鍋囨渚?
    fn = np.zeros(num_classes)  # 鍋囪礋渚?
    tn = np.zeros(num_classes)  # 鐪熻礋渚?
    support = np.zeros(num_classes)  # 姣忕被鐪熷疄鏍锋湰鏁?

    # 缁熻TP/FP/FN/TN
    for class_id in range(num_classes):
        # 鐪熷疄涓篶lass_id鐨勬牱鏈暟
        support[class_id] = np.sum(y_true == class_id)
        # TP: 鐪熷疄鏄痗lass_id锛岄娴嬩篃鏄痗lass_id
        tp[class_id] = np.sum((y_true == class_id) & (y_pred == class_id))
        # FP: 鐪熷疄涓嶆槸class_id锛岄娴嬫槸class_id
        fp[class_id] = np.sum((y_true != class_id) & (y_pred == class_id))
        # FN: 鐪熷疄鏄痗lass_id锛岄娴嬩笉鏄痗lass_id
        fn[class_id] = np.sum((y_true == class_id) & (y_pred != class_id))
        # TN: 鐪熷疄涓嶆槸class_id锛岄娴嬩篃涓嶆槸class_id
        tn[class_id] = np.sum((y_true != class_id) & (y_pred != class_id))

    # 璁＄畻姣忕被鎸囨爣
    valid_classes = 0  # 鏈夋牱鏈殑绫诲埆鏁?
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for class_id in range(num_classes):
        metrics["support"][class_id] = int(support[class_id])

        if support[class_id] == 0:
            # 鏃犳牱鏈殑绫诲埆锛屾寚鏍囪涓?
            metrics["per_class"][class_id] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }
            continue

        # 绮剧‘鐜?= TP / (TP + FP)
        precision = tp[class_id] / (tp[class_id] + fp[class_id]) if (tp[class_id] + fp[class_id]) > 0 else 0.0
        # 鍙洖鐜?= TP / (TP + FN)
        recall = tp[class_id] / (tp[class_id] + fn[class_id]) if (tp[class_id] + fn[class_id]) > 0 else 0.0
        # F1鍒嗘暟 = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics["per_class"][class_id] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

        # 绱姞鐢ㄤ簬瀹忓钩鍧?
        metrics["macro_avg"]["precision"] += precision
        metrics["macro_avg"]["recall"] += recall
        metrics["macro_avg"]["f1"] += f1
        valid_classes += 1

        # 绱姞鐢ㄤ簬寰钩鍧?
        total_tp += tp[class_id]
        total_fp += fp[class_id]
        total_fn += fn[class_id]

    # 璁＄畻瀹忓钩鍧囷紙鏈夋牱鏈被鍒殑骞冲潎锛?
    if valid_classes > 0:
        metrics["macro_avg"]["precision"] = round(metrics["macro_avg"]["precision"] / valid_classes, 4)
        metrics["macro_avg"]["recall"] = round(metrics["macro_avg"]["recall"] / valid_classes, 4)
        metrics["macro_avg"]["f1"] = round(metrics["macro_avg"]["f1"] / valid_classes, 4)

    # 璁＄畻寰钩鍧囷紙鍏ㄥ眬鎸囨爣锛?
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    metrics["micro_avg"]["precision"] = round(micro_precision, 4)
    metrics["micro_avg"]["recall"] = round(micro_recall, 4)
    metrics["micro_avg"]["f1"] = round(micro_f1, 4)

    return metrics

def plot_confusion_matrix(cfg: dict, y_true, y_pred, class_names=None):
    """
    绾墜鍔ㄧ粯鍒舵贩娣嗙煩闃碉紙鏃爏eaborn/sklearn渚濊禆锛?
    """
    print(f"\n[*] Generating confusion matrix (manual calculation)...")

    # 璁剧疆绫诲埆鍚嶇О
    num_classes = cfg["num_action_ids"]
    if class_names is None:
        class_names = [f"Action {i}" for i in range(num_classes)]

    # 鎵嬪姩璁＄畻娣锋穯鐭╅樀
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)

    # 鎵嬪姩褰掍竴鍖?
    if cfg["cm_normalize"]:
        cm_plot = normalize_confusion_matrix(cm)
        fmt = ".2f"
        title_suffix = "(Normalized)"
    else:
        cm_plot = cm
        fmt = "d"
        title_suffix = ""

    # 鍒涘缓鐢诲竷
    fig, ax = plt.subplots(figsize=cfg["cm_figsize"])

    # 缁樺埗鐑姏鍥撅紙绾痬atplotlib瀹炵幇锛?
    im = ax.imshow(cm_plot, cmap="Blues", aspect="auto")

    # 娣诲姞棰滆壊鏉?
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count" if not cfg["cm_normalize"] else "Normalized Value", rotation=-90, va="bottom")

    # 璁剧疆鍒诲害鏍囩
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, fontsize=cfg["cm_fontsize"])
    ax.set_yticklabels(class_names, fontsize=cfg["cm_fontsize"])

    # 鏃嬭浆x杞存爣绛?
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 鍦ㄦ瘡涓崟鍏冩牸涓坊鍔犳枃鏈?
    for i in range(num_classes):
        for j in range(num_classes):
            # 閫夋嫨鏂囨湰棰滆壊锛堟牴鎹儗鏅壊娣辨祬锛?
            text_color = "white" if cm_plot[i, j] > (cm_plot.max() / 2) else "black"
            # 鏍煎紡鍖栨枃鏈?
            text = ax.text(j, i, format(cm_plot[i, j], fmt),
                           ha="center", va="center", color=text_color, fontsize=cfg["cm_fontsize"])

    # 璁剧疆鏍囬鍜屾爣绛?
    ax.set_title(f"Confusion Matrix {title_suffix}", fontsize=cfg["cm_fontsize"] + 4, pad=20)
    ax.set_xlabel("Predicted Label", fontsize=cfg["cm_fontsize"] + 2, labelpad=15)
    ax.set_ylabel("True Label", fontsize=cfg["cm_fontsize"] + 2, labelpad=15)

    # 璋冩暣甯冨眬
    fig.tight_layout()

    # 淇濆瓨鍥剧墖
    cm_filename = f"confusion_matrix_normalized_{cfg['cm_normalize']}.{cfg['cm_save_format']}"
    cm_path = os.path.join(cfg["output_dir"], cm_filename)
    plt.savefig(cm_path, dpi=cfg["cm_dpi"], bbox_inches='tight')
    plt.close()

    print(f"[*] Confusion matrix saved to: {cm_path}")

    # 棰濆淇濆瓨鍙鐢ㄧ殑娣锋穯鐭╅樀鏁版嵁锛堢敤浜庤法鏁版嵁闆嗚瀺鍚堬級
    cm_raw_path = os.path.join(cfg["output_dir"], "confusion_matrix_raw_counts.npy")
    cm_norm_path = os.path.join(cfg["output_dir"], "confusion_matrix_row_normalized.npy")
    np.save(cm_raw_path, cm)
    np.save(cm_norm_path, cm_plot)

    cm_json_path = os.path.join(cfg["output_dir"], "confusion_matrix_data.json")
    with open(cm_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "class_names": class_names,
                "raw_counts": cm.tolist(),
                "row_normalized": cm_plot.tolist(),
                "is_normalized_figure": bool(cfg["cm_normalize"]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[*] Confusion matrix data saved to: {cm_raw_path}, {cm_norm_path}, {cm_json_path}")

    # 鎵嬪姩璁＄畻鍒嗙被鎶ュ憡
    metrics = compute_classification_metrics(y_true, y_pred, num_classes)

    # 鎵撳嵃鍒嗙被鎶ュ憡
    print("\n[*] Classification Report (Manual Calculation):")
    print("="*100)
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-"*100)
    for class_id in range(num_classes):
        cls_metrics = metrics["per_class"][class_id]
        support = metrics["support"][class_id]
        print(f"{class_names[class_id]:<10} {cls_metrics['precision']:<10.4f} {cls_metrics['recall']:<10.4f} {cls_metrics['f1']:<10.4f} {support:<10d}")
    print("-"*100)
    print(f"{'Macro Avg':<10} {metrics['macro_avg']['precision']:<10.4f} {metrics['macro_avg']['recall']:<10.4f} {metrics['macro_avg']['f1']:<10.4f} {sum(metrics['support'].values()):<10d}")
    print(f"{'Micro Avg':<10} {metrics['micro_avg']['precision']:<10.4f} {metrics['micro_avg']['recall']:<10.4f} {metrics['micro_avg']['f1']:<10.4f} {sum(metrics['support'].values()):<10d}")
    print("="*100)

    # 淇濆瓨鍒嗙被鎶ュ憡
    report_path = os.path.join(cfg["output_dir"], "classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Classification Report (Manual Calculation)\n")
        f.write("="*100 + "\n")
        f.write(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
        f.write("-"*100 + "\n")
        for class_id in range(num_classes):
            cls_metrics = metrics["per_class"][class_id]
            support = metrics["support"][class_id]
            f.write(f"{class_names[class_id]:<10} {cls_metrics['precision']:<10.4f} {cls_metrics['recall']:<10.4f} {cls_metrics['f1']:<10.4f} {support:<10d}\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Macro Avg':<10} {metrics['macro_avg']['precision']:<10.4f} {metrics['macro_avg']['recall']:<10.4f} {metrics['macro_avg']['f1']:<10.4f} {sum(metrics['support'].values()):<10d}\n")
        f.write(f"{'Micro Avg':<10} {metrics['micro_avg']['precision']:<10.4f} {metrics['micro_avg']['recall']:<10.4f} {metrics['micro_avg']['f1']:<10.4f} {sum(metrics['support'].values()):<10d}\n")
        f.write("="*100 + "\n")

    print(f"[*] Classification report saved to: {report_path}")

    return cm

# ==============================================================================
# 3. 鍘熸湁杈呭姪鍑芥暟锛堟棤淇敼锛?
# ==============================================================================

def _compute_error_run_lengths(error_flags: np.ndarray) -> list:
    run_lengths = []
    current = 0
    for flag in error_flags:
        if flag:
            current += 1
        elif current > 0:
            run_lengths.append(current)
            current = 0
    if current > 0:
        run_lengths.append(current)
    return run_lengths


def _compute_boundary_mask(labels: np.ndarray, tolerance: int) -> np.ndarray:
    n = len(labels)
    mask = np.zeros(n, dtype=bool)
    if n <= 1:
        return mask
    change_points = np.where(labels[1:] != labels[:-1])[0] + 1
    for cp in change_points:
        left = max(0, cp - tolerance)
        right = min(n, cp + tolerance + 1)
        mask[left:right] = True
    return mask


def save_temporal_error_analysis(cfg: dict, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if not cfg.get("enable_temporal_error_analysis", True):
        return {}

    n = len(y_true)
    if n == 0:
        return {}

    err = (y_true != y_pred)
    tol = int(cfg.get("temporal_boundary_tolerance", 2))
    boundary_mask = _compute_boundary_mask(y_true, tol)

    non_boundary_mask = ~boundary_mask
    non_boundary_total = int(non_boundary_mask.sum())
    non_boundary_errors = int((err & non_boundary_mask).sum())
    if non_boundary_total > 0:
        boundary_tolerant_acc = float(1.0 - non_boundary_errors / non_boundary_total)
    else:
        boundary_tolerant_acc = 1.0

    run_lengths = _compute_error_run_lengths(err.astype(np.int32))
    mean_error_run = float(np.mean(run_lengths)) if len(run_lengths) > 0 else 0.0
    max_error_run = int(np.max(run_lengths)) if len(run_lengths) > 0 else 0
    p95_error_run = int(np.percentile(run_lengths, 95)) if len(run_lengths) > 0 else 0

    L = int(cfg.get("catastrophic_error_run_threshold", 8))
    catastrophic_runs = [r for r in run_lengths if r >= L]
    if len(run_lengths) > 0:
        catastrophic_run_rate = float(len(catastrophic_runs) / len(run_lengths))
    else:
        catastrophic_run_rate = 0.0

    segment_fail_threshold = float(cfg.get("segment_failure_ratio_threshold", 0.6))
    total_error_frames = int(err.sum())
    catastrophic_error_frames = int(sum(catastrophic_runs))
    if len(run_lengths) > 0:
        segment_failure_rate = float(catastrophic_error_frames / total_error_frames)
    else:
        segment_failure_rate = 0.0

    metrics = {
        "num_frames": int(n),
        "boundary_tolerance": tol,
        "boundary_tolerant_accuracy": boundary_tolerant_acc,
        "num_error_segments": int(len(run_lengths)),
        "mean_error_run_length": mean_error_run,
        "p95_error_run_length": p95_error_run,
        "max_error_run_length": max_error_run,
        "catastrophic_run_threshold": L,
        "catastrophic_run_rate": catastrophic_run_rate,
        "segment_failure_ratio_threshold": segment_fail_threshold,
        "segment_failure_rate": segment_failure_rate,
    }

    save_path = os.path.join(cfg["output_dir"], "temporal_error_report.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n[*] Temporal error analysis:")
    print(f"    Boundary-tolerant@{tol} accuracy: {metrics['boundary_tolerant_accuracy']:.4f}")
    print(f"    Mean error run length: {metrics['mean_error_run_length']:.3f}")
    print(f"    Catastrophic run rate (>= {L}): {metrics['catastrophic_run_rate']:.4f}")
    print(f"    Saved to: {save_path}")

    return metrics


def _build_action_to_stage_map(stage_definitions: dict) -> dict:
    action_to_stage = {}
    for stage_id, action_ids in stage_definitions.items():
        for action_id in action_ids:
            action_to_stage[int(action_id)] = int(stage_id)
    return action_to_stage


def save_mechanism_metrics(cfg: dict, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """淇濆瓨鐢ㄤ簬鏈哄埗鍒嗘瀽鐨?stage-level 鎸囨爣锛堟棤闇€閲嶈窇璁粌锛夈€?""
    stage_definitions = cfg.get("stage_definitions", {})
    if not stage_definitions:
        return {}

    action_to_stage = _build_action_to_stage_map(stage_definitions)

    valid_mask = np.array([(int(y) in action_to_stage) for y in y_true], dtype=bool)
    if valid_mask.sum() == 0:
        return {}

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    true_stage = np.array([action_to_stage[int(y)] for y in y_true_valid], dtype=np.int64)
    pred_stage = np.array([action_to_stage[int(y)] for y in y_pred_valid], dtype=np.int64)

    num_stages = len(stage_definitions)
    stage_cm = compute_confusion_matrix(true_stage, pred_stage, num_stages)
    stage_acc = float(np.mean(true_stage == pred_stage)) if len(true_stage) > 0 else 0.0

    action_conflicts = int(np.sum(y_true_valid != y_pred_valid))
    stage_conflicts = int(np.sum((y_true_valid != y_pred_valid) & (true_stage != pred_stage)))
    within_stage_conflicts = int(np.sum((y_true_valid != y_pred_valid) & (true_stage == pred_stage)))
    if action_conflicts > 0:
        conflict_cross_stage_ratio = float(stage_conflicts / action_conflicts)
    else:
        conflict_cross_stage_ratio = 0.0

    support_per_stage = {}
    for stage_id in sorted(stage_definitions.keys()):
        support_per_stage[int(stage_id)] = int(np.sum(true_stage == int(stage_id)))

    metrics = {
        "stage_accuracy": stage_acc,
        "num_stages": num_stages,
        "stage_confusion_matrix": stage_cm.tolist(),
        "support_per_stage": support_per_stage,
        "num_action_misclassifications": action_conflicts,
        "num_cross_stage_misclassifications": stage_conflicts,
        "num_within_stage_misclassifications": within_stage_conflicts,
        "cross_stage_misclassification_ratio": conflict_cross_stage_ratio,
    }

    save_path = os.path.join(cfg["output_dir"], "mechanism_metrics.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n[*] Mechanism metrics:")
    print(f"    Stage accuracy: {metrics['stage_accuracy']:.4f}")
    print(f"    Cross-stage misclassification ratio: {metrics['cross_stage_misclassification_ratio']:.4f}")
    print(f"    Saved to: {save_path}")

    return metrics


def setup_environment(cfg: dict) -> torch.device:
    """璁剧疆楠岃瘉鐜锛屽寘鎷澶囬€夋嫨鍜岀洰褰曞垱寤恒€?""
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    return device

def create_models(cfg: dict, device: torch.device) -> tuple[
    nn.Module, nn.Module, nn.Module, nn.Module, PrismaticProcessor]:
    """鍒涘缓骞跺姞杞芥墍鏈夋ā鍨嬬粍浠讹紙涓庤缁冭剼鏈繚鎸佷竴鑷达級"""
    print("\n[*] Creating models...")

    # 娉ㄥ唽OpenVLA妯″瀷
    if model_is_on_hf_hub(cfg["config_path"]):
        vla_download_path = snapshot_download(repo_id=cfg["config_path"])
        cfg["config_path"] = vla_download_path
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    processor = AutoProcessor.from_pretrained(cfg["config_path"], trust_remote_code=True)

    # 鍔犺浇VLM妯″瀷
    print(f"[*] Loading VLM model from: {cfg['vlm_path']}")
    hf_token = ''
    if 'prism-qwen25-extra-dinosiglip-224px-0_5b' in cfg['vlm_path']:
        vlm = load(cfg["vlm_path"], hf_token=hf_token, load_for_training=False)
    else:
        vlm = load_vla(
            cfg['vlm_path'],
            hf_token=hf_token,
            load_for_training=False,
        )

    # 鍔犺浇VLA閰嶇疆
    print("[*] Loading VLA config from: <PATH_TO_VLA_CONFIG>/config.json")
    vla_config = AutoConfig.from_pretrained("<PATH_TO_VLA_CONFIG>/config.json")

    # 鍒涘缓VLA妯″瀷
    print("[*] Creating VLA model from config...")
    vla = AutoModelForVision2Seq.from_config(vla_config, torch_dtype=torch.bfloat16).to(device)

    # 鏉冮噸閿悕鏇挎崲
    replace_map = [
        ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
        ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
        ("llm_backbone.llm", "language_model"),
        ("projector.projector.0", "projector.fc1"),
        ("projector.projector.2", "projector.fc2"),
        ("projector.projector.4", "projector.fc3"),
        ("gamma", "scale_factor"),
    ]

    def rename_state_dict_keys(state_dict, replace_map):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            for old, new in replace_map:
                if old in new_k:
                    new_k = new_k.replace(old, new)
            new_state_dict[new_k] = v
        return new_state_dict

    # 杞崲鏉冮噸
    print("[*] Renaming VLM state dict keys...")
    old_state_dict = vlm.state_dict()
    new_state_dict = rename_state_dict_keys(old_state_dict, replace_map)
    del old_state_dict, vlm

    # 鍔犺浇鏉冮噸鍒癡LA
    print("[*] Loading converted weights into VLA model...")
    missing_keys, unexpected_keys = vla.load_state_dict(new_state_dict, strict=False)
    print(f"    - Missing keys: {len(missing_keys)}")
    print(f"    - Unexpected keys: {len(unexpected_keys)}")

    # 璁剧疆杈撳叆鍥惧儚鏁伴噺
    if "num_images_in_input" in cfg:
        print(f"[*] Setting number of images in input to: {cfg['num_images_in_input']}")
        vla.vision_backbone.set_num_images_in_input(cfg["num_images_in_input"])

    # 鑾峰彇LLM缁村害
    llm_dim = vla.module.llm_dim if hasattr(vla, 'module') else vla.llm_dim
    print(f"[*] LLM dimension detected as: {llm_dim}")

    # 鍒涘缓Action Head鍜孭roprio Projector
    print("[*] Creating standalone action head and proprio projector...")
    if cfg["use_moe_action_head"]:
        action_head = StageMoEActionHead(
            input_dim=llm_dim,
            hidden_dim=llm_dim,
            action_dim=cfg.get("ACTION_DIM", 8),
        ).to(device)
    else:
        action_head = L1RegressionActionHead(
            input_dim=llm_dim,
            hidden_dim=llm_dim,
            action_dim=cfg.get("ACTION_DIM", 8),
        ).to(device)

    proprio_projector = ProprioProjector(
        llm_dim=llm_dim,
        proprio_dim=cfg.get("proprio_dim", 8),
    ).to(device)

    # 鍒涘缓鍒ゅ埆鍣?
    print("[*] Creating Action ID Discriminator...")
    vision_backbone = DinoSigLIPViTBackbone(
        vision_backbone_id=cfg["vision_backbone_id"],
        image_resize_strategy=cfg["image_resize_strategy"],
        default_image_size=cfg["image_size"],
    ).to(device)

    # 鍐荤粨瑙嗚楠ㄥ共缃戠粶
    for param in vision_backbone.parameters():
        param.requires_grad = False

    discriminator = ActionIDDiscriminator(
        num_action_ids=cfg["num_action_ids"],
        vision_backbone=vision_backbone,
        llm_dim=llm_dim,
        proprio_dim=cfg.get("proprio_dim", 8),
        hidden_dim=cfg.get("discriminator_hidden_dim", 512)
    ).to(device)
    discriminator = discriminator.to(torch.bfloat16)

    # 鍔犺浇鍒ゅ埆鍣ㄦ潈閲?
    print(f"[*] Loading discriminator checkpoint from: {cfg['checkpoint_path']}")
    checkpoint = torch.load(cfg["checkpoint_path"], map_location=device)

    # ===== 鍏煎涓ょ鏍煎紡 =====
    # 1) 鏂版牸寮忥細{"discriminator_state_dict": ...}
    # 2) 鏃ф牸寮忥細鐩存帴鏄?state_dict锛堜綘鐜板湪鐨勶級
    if isinstance(checkpoint, dict) and "discriminator_state_dict" in checkpoint:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        print("[*] Discriminator weights loaded from wrapped checkpoint.")
    else:
        # 鐩存帴褰撲綔瑁?state_dict 鍔犺浇
        discriminator.load_state_dict(checkpoint)
        print("[*] Discriminator weights loaded from raw state_dict.")

    print("[*] Models created successfully.\n")
    return vla, action_head, proprio_projector, discriminator, processor

def create_data_loader(cfg: dict, processor: PrismaticProcessor) -> DataLoader:
    """鍒涘缓楠岃瘉鏁版嵁鍔犺浇鍣?""
    print("[*] Creating validation data loader...")
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=True,
        use_proprio=True,
        use_minivlm=True
    )

    resize_resolution = (cfg["image_size"], cfg["image_size"])

    val_dataset = RLDSDataset(
        data_root_dir=cfg["data_root_dir"], data_mix=cfg["dataset_name"],
        batch_transform=batch_transform, resize_resolution=resize_resolution,
        shuffle_buffer_size=cfg.get("shuffle_buffer_size", 10_000), image_aug=False, train=cfg.get("eval_train_split", True)
    )

    collator = PaddedCollatorForActionPrediction(
        model_max_length=processor.tokenizer.model_max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
        padding_side="right"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    print(f"[*] Validation dataset size: {len(val_dataset)}\n")
    return val_loader

# ==============================================================================
# 4. 涓婚獙璇佸嚱鏁帮紙鏃犱慨鏀癸紝浠呰皟鐢ㄦ柊鐨勬贩娣嗙煩闃靛嚱鏁帮級
# ==============================================================================

def validate(cfg: dict):
    """鎵ц瀹屾暣鐨勯獙璇佹祦绋?""
    # 1. 鐜璁剧疆
    device = setup_environment(cfg)

    # 2. 鍒涘缓妯″瀷
    vla, action_head, proprio_projector, discriminator, processor = create_models(cfg, device)

    # 3. 鑾峰彇num_patches
    num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    print(f"[*] Number of patches: {num_patches}")

    # 4. 鍒涘缓鏁版嵁鍔犺浇鍣?
    val_loader = create_data_loader(cfg, processor)

    # 5. 鍑嗗楠岃瘉
    discriminator.eval()
    vla.eval()
    action_head.eval()
    proprio_projector.eval()

    loss_fn = ActionIDLoss()

    # 瀛樺偍鎵€鏈夐娴嬬粨鏋滃拰鐪熷疄鏍囩
    all_preds = []
    all_labels = []
    all_losses = []
    skipped_no_action_id_batches = 0

    val_start_time = time.time()
    progress_bar = tqdm.tqdm(val_loader, desc="Validating")

    # 6. 楠岃瘉寰幆
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar, start=1):
            # 妫€鏌ヨ秴鏃?
            if cfg["val_time_limit"] is not None and (time.time() - val_start_time > cfg["val_time_limit"]):
                print(f"\n[!] Validation time limit ({cfg['val_time_limit']}s) exceeded. Stopping early.")
                break

            # 鏁版嵁鍑嗗
            if cfg.get("max_batches") is not None and batch_idx > int(cfg["max_batches"]):
                print(f"\n[!] Reached max_batches={cfg['max_batches']}. Stopping early.")
                break
            pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device)
            proprio = batch["proprio"].to(device)
            # import pdb;pdb.set_trace()
            raw_action_ids = batch.get("action_id", batch.get("action_ids", None))
            if raw_action_ids is None:
                skipped_no_action_id_batches += 1
                if skipped_no_action_id_batches == 1:
                    print(f"[!] Batch has no action_id/action_ids. Keys: {list(batch.keys())}")
                continue
            action_ids = raw_action_ids.squeeze().to(device)
            labels = batch.get("labels").to(device)
            input_ids = batch.get("input_ids").to(device)
            attention_mask = batch.get("attention_mask").to(device)

            # VLA鍓嶅悜浼犳挱
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                    output_hidden_states=True,
                    proprio=True,
                    proprio_projector=True,
                    noisy_actions=None,
                    noisy_action_projector=None,
                    diffusion_timestep_embeddings=None,
                    use_film=False
                )

            # 鐗瑰緛鎻愬彇
            ground_truth_token_ids = batch["labels"][:, 1:].to(device)
            current_action_mask = get_current_action_mask(ground_truth_token_ids)
            next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

            task_features = output.hidden_states[-1][:, :num_patches, :].to(torch.bfloat16)
            text_hidden_states = output.hidden_states[-1][:, num_patches:-1, :]
            bs = text_hidden_states.shape[0]
            action_features = text_hidden_states[current_action_mask | next_actions_mask].reshape(
                bs, -1, text_hidden_states.shape[-1]
            ).to(torch.bfloat16)

            # 鍒ゅ埆鍣ㄥ墠鍚戜紶鎾?
            action_probs, confidence, logits = discriminator(
                pixel_values=pixel_values,
                proprio=proprio,
                input_ids=input_ids,
                task_features=task_features,
                action_features=action_features
            )

            # 璁＄畻鎹熷け鍜屽噯纭巼
            # loss = loss_fn(logits,action_ids,confidence)
            preds = logits.argmax(dim=-1)

            # 鏀堕泦缁撴灉
            # all_losses.append(loss.item())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(action_ids.cpu().numpy())

            # 鏇存柊杩涘害鏉?
            current_acc = (preds == action_ids).float().mean().item()
            progress_bar.set_postfix({
                # "loss": f"{loss.item():.4f}",
                "acc": f"{current_acc:.4f}",
                "elapsed": f"{time.time() - val_start_time:.1f}s"
            })

    # 7. 璁＄畻鏁翠綋鎸囨爣
    avg_loss = np.mean(all_losses) if all_losses else 0.0
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if skipped_no_action_id_batches > 0:
        print(f"[!] Skipped {skipped_no_action_id_batches} batches due to missing action_id/action_ids")
    if len(all_labels) == 0:
        raise RuntimeError(
            "No action_id found in validation batches. "
            "Please check dataset fields or collator output keys (expected action_id/action_ids)."
        )

    overall_accuracy = np.mean(all_preds == all_labels)

    # 8. 鎵撳嵃鎬讳綋缁撴灉
    print("\n" + "="*80)
    print("                      Validation Results                      ")
    print("="*80)
    print(f"Checkpoint: {cfg['checkpoint_path']}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({100*overall_accuracy:.2f}%)")
    print(f"Total Samples: {len(all_labels)}")
    print(f"Validation Time: {time.time() - val_start_time:.1f}s")
    print("="*80)

    # 9. 璁＄畻姣忕被鍑嗙‘鐜?
    print("\n[*] Per-class Accuracy:")
    class_accuracies = {}
    for class_id in range(cfg["num_action_ids"]):
        mask = all_labels == class_id
        if np.sum(mask) > 0:
            acc = np.mean(all_preds[mask] == all_labels[mask])
            class_accuracies[class_id] = acc
            print(f"  Action {class_id}: {acc:.4f} (n={np.sum(mask)})")
        else:
            class_accuracies[class_id] = 0.0
            print(f"  Action {class_id}: N/A (no samples)")

    # 10. 鐢熸垚娣锋穯鐭╅樀锛堢函鎵嬪姩瀹炵幇锛?
    plot_confusion_matrix(cfg, all_labels, all_preds)

    # 10.1 鏈哄埗鍒嗘瀽锛坰tage-level锛?
    mechanism_metrics = save_mechanism_metrics(cfg, all_labels, all_preds)
    temporal_metrics = save_temporal_error_analysis(cfg, all_labels, all_preds)

    # 11. 淇濆瓨楠岃瘉缁撴灉
    results = {
        "checkpoint_path": cfg["checkpoint_path"],
        "average_loss": avg_loss,
        "overall_accuracy": overall_accuracy,
        "total_samples": len(all_labels),
        "validation_time": time.time() - val_start_time,
        "class_accuracies": class_accuracies,
        "mechanism_metrics": mechanism_metrics,
        "temporal_metrics": temporal_metrics,
        "num_action_ids": cfg["num_action_ids"],
        "val_config": cfg
    }

    results_path = os.path.join(cfg["output_dir"], "validation_results.npy")
    np.save(results_path, results)
    print(f"\n[*] Validation results saved to: {results_path}")

    return {
        "avg_loss": avg_loss,
        "overall_accuracy": overall_accuracy,
        "class_accuracies": class_accuracies
    }

# ==============================================================================
# 5. 鑴氭湰鍏ュ彛
# ==============================================================================
import copy
import csv


def parse_args():
    parser = argparse.ArgumentParser(description="Validate an Action ID discriminator checkpoint.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset name, e.g. libero_bread_action_id")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to discriminator checkpoint")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save evaluation artifacts")
    parser.add_argument("--data-root-dir", type=str, default=None, help="Root directory that contains the RLDS dataset")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Which split to evaluate")
    parser.add_argument("--batch-size", type=int, default=None, help="Validation batch size")
    parser.add_argument("--val-time-limit", type=int, default=None, help="Optional validation time limit in seconds")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional hard stop after a fixed number of batches")
    parser.add_argument("--run-batch-suite", action="store_true", help="Run the original 4-dataset batch evaluation suite")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not args.run_batch_suite:
        cfg = copy.deepcopy(CONFIG)
        if args.dataset_name is not None:
            cfg["dataset_name"] = args.dataset_name
        if args.checkpoint_path is not None:
            cfg["checkpoint_path"] = args.checkpoint_path
        if args.output_dir is not None:
            cfg["output_dir"] = args.output_dir
        else:
            ckpt_name = Path(cfg["checkpoint_path"]).stem
            cfg["output_dir"] = os.path.join(CONFIG["output_dir"], f"{cfg['dataset_name']}_{args.split}_{ckpt_name}")
        if args.data_root_dir is not None:
            cfg["data_root_dir"] = args.data_root_dir
        if args.batch_size is not None:
            cfg["batch_size"] = args.batch_size
        cfg["eval_train_split"] = (args.split == "train")
        if args.val_time_limit is not None:
            cfg["val_time_limit"] = args.val_time_limit
        cfg["max_batches"] = args.max_batches

        print("\n" + "=" * 100)
        print(f"Running single evaluation: dataset={cfg['dataset_name']}, split={args.split}")
        print(f"Checkpoint: {cfg['checkpoint_path']}")
        print(f"Output dir: {cfg['output_dir']}")
        print("=" * 100)
        validate(cfg)
        raise SystemExit(0)
    # 浣犺璇勪及鐨?4 涓瓙鏁版嵁闆?
    dataset_list = [
        "libero_object_action_id",
        "libero_goal_action_id",
        "libero_spatial_action_id",
        "libero_10_action_id",  # 濡傛灉杩欐槸浣犵殑 long 瀛愰泦鍚?
    ]

    # train/val 閮藉彲閫夛紱濡傛灉鏌愪簺鏁版嵁闆嗘病鏈?val split锛屽厛鐢?train
    split_list = [True]   # True=train, False=val
    # split_list = [True, False]  # 鎯充袱涓兘璺戝氨寮€杩欎釜

    timeout_map = {
        "libero_object_action_id": 900,
        "libero_goal_action_id": 900,
        "libero_spatial_action_id": 900,
        "libero_10_action_id": 1500,
    }

    all_rows = []

    for ds_name in dataset_list:
        for is_train in split_list:
            cfg = copy.deepcopy(CONFIG)
            cfg["dataset_name"] = ds_name
            cfg["val_time_limit"] = timeout_map.get(ds_name, 900)  # 姣忎釜瀛愭暟鎹泦杩愯 15 鍒嗛挓
            cfg["output_dir"] = os.path.join(
                CONFIG["output_dir"],
                f"{ds_name}_{'train' if is_train else 'val'}"
            )
            cfg["eval_train_split"] = is_train  # 涓嬮潰 create_data_loader 浼氱敤鍒?

            print("\n" + "=" * 100)
            print(f"Running: dataset={ds_name}, split={'train' if is_train else 'val'}")
            print("=" * 100)

            result = validate(cfg)

            mech_path = os.path.join(cfg["output_dir"], "mechanism_metrics.json")
            temporal_path = os.path.join(cfg["output_dir"], "temporal_error_report.json")
            mech_metrics = {}
            temporal_metrics = {}
            if os.path.exists(mech_path):
                with open(mech_path, "r", encoding="utf-8") as f:
                    mech_metrics = json.load(f)
            if os.path.exists(temporal_path):
                with open(temporal_path, "r", encoding="utf-8") as f:
                    temporal_metrics = json.load(f)

            row = {
                "dataset": ds_name,
                "split": "train" if is_train else "val",
                "top1_acc": float(result["overall_accuracy"]),
                "avg_loss": float(result["avg_loss"]),
                "stage_acc": float(mech_metrics.get("stage_accuracy", 0.0)),
                "cross_stage_err_ratio": float(mech_metrics.get("cross_stage_misclassification_ratio", 0.0)),
                "boundary_tolerant_acc": float(temporal_metrics.get("boundary_tolerant_accuracy", 0.0)),
                "mean_error_run_len": float(temporal_metrics.get("mean_error_run_length", 0.0)),
                "catastrophic_run_rate": float(temporal_metrics.get("catastrophic_run_rate", 0.0)),
                "output_dir": cfg["output_dir"],
            }
            all_rows.append(row)

    # 姹囨€讳繚瀛?
    summary_csv = os.path.join(CONFIG["output_dir"], "summary_top1_4subsets.csv")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "split", "top1_acc", "avg_loss", "stage_acc", "cross_stage_err_ratio", "boundary_tolerant_acc", "mean_error_run_len", "catastrophic_run_rate", "output_dir"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nAll runs finished.")
    print(f"Summary saved to: {summary_csv}")

