"""Evaluation metrics for crack segmentation and geometric quantification.

Provides pixel-level segmentation metrics (IoU, F1/Dice, Precision, Recall),
boundary-tolerant F1 for thin structures, and regression error metrics
(MAE, RMSE, MAPE) for crack length and width estimation.
"""

"""
- TP: 실제 균열(1)을 모델도 균열(1)로 맞힌 픽셀 수
- FP: 배경(0)인데 모델은 균열(1)로 찍은 픽셀 수
- FN: 실제 균열(1)인데 모델은 배경(0)으로 찍은 픽셀 수

Precision: (TP) / (TP+FP) 모델이 균열로 예측한 것 중 진짜 균열 비율
Recall: (TP) / (TP+FN) 전체 균열 중 모델이 찾아낸 비율
F1-score: (2TP) / (2TP+FP+FN) precision, Recall의 조화평균
IoU: (TP) / (TP+FP+FM) 교집합 / 합집합

* 분모 0 되는 것을 방지하기 위해 1e-7 더해줌
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Union

import cv2
import numpy as np

EPSILON = 1e-7


def _to_binary_uint8(mask: np.ndarray) -> np.ndarray:
    """Convert any 2D mask or probability map (thresholded at > 0) to uint8 {0, 1}."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr.squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {arr.shape}")
    return (arr > 0).astype(np.uint8)


def compute_confusion_counts(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> Dict[str, int]:
    """Compute pixel-wise True Positive, False Positive, and False Negative counts."""
    pred = _to_binary_uint8(pred_mask)
    gt = _to_binary_uint8(gt_mask)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    tp = int(np.logical_and(pred == 1, gt == 1).sum())
    fp = int(np.logical_and(pred == 1, gt == 0).sum())
    fn = int(np.logical_and(pred == 0, gt == 1).sum())
    return {"tp": tp, "fp": fp, "fn": fn}


def metrics_from_counts(tp: float, fp: float, fn: float) -> Dict[str, float]:
    """Compute Precision, Recall, F1 (Dice), and IoU from aggregated counts."""
    if tp == 0 and fp == 0 and fn == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0}

    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + EPSILON)
    iou = tp / (tp + fp + fn + EPSILON)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


# Tolerance F1: 예측한 픽셀이 정답과 n픽셀 오차 내에 위치하면 정답으로 인정
def compute_tolerance_counts(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_px: int = 2,
) -> Dict[str, int]:
    
    pred = _to_binary_uint8(pred_mask)
    gt = _to_binary_uint8(gt_mask)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    # 음수 입력 방지
    if tolerance_px <= 0:
        counts = compute_confusion_counts(pred, gt)
        return {
            "tol_tp_pred": counts["tp"],
            "pred_total": counts["tp"] + counts["fp"],
            "tol_tp_gt": counts["tp"],
            "gt_total": counts["tp"] + counts["fn"],
        }

    pred_total = int(pred.sum())
    gt_total = int(gt.sum())

    if pred_total == 0 or gt_total == 0:
        return {
            "tol_tp_pred": 0,
            "pred_total": pred_total,
            "tol_tp_gt": 0,
            "gt_total": gt_total,
        }

    # Distance from every pixel to the nearest foreground pixel (where 1-mask == 0)
    dist_to_gt = cv2.distanceTransform(1 - gt, cv2.DIST_L2, 5)
    dist_to_pred = cv2.distanceTransform(1 - pred, cv2.DIST_L2, 5)

    tol_tp_pred = int(np.logical_and(pred == 1, dist_to_gt <= tolerance_px).sum())
    tol_tp_gt = int(np.logical_and(gt == 1, dist_to_pred <= tolerance_px).sum())

    return {
        "tol_tp_pred": tol_tp_pred,
        "pred_total": pred_total,
        "tol_tp_gt": tol_tp_gt,
        "gt_total": gt_total,
    }


def tolerance_metrics_from_counts(
    tol_tp_pred: float,
    pred_total: float,
    tol_tp_gt: float,
    gt_total: float,
) -> Dict[str, float]:
    """Compute boundary-tolerant Precision, Recall, and F1 from aggregated counts."""
    if pred_total == 0 and gt_total == 0:
        return {"tol_precision": 1.0, "tol_recall": 1.0, "tol_f1": 1.0}

    tol_precision = tol_tp_pred / (pred_total + EPSILON)
    tol_recall = tol_tp_gt / (gt_total + EPSILON)
    tol_f1 = (2.0 * tol_precision * tol_recall) / (tol_precision + tol_recall + EPSILON)
    return {
        "tol_precision": float(tol_precision),
        "tol_recall": float(tol_recall),
        "tol_f1": float(tol_f1),
    }


def evaluate_pair(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_px: int = 2,
) -> Dict[str, float]:
    """Evaluate a single (prediction, ground-truth) binary mask pair."""
    c = compute_confusion_counts(pred_mask, gt_mask)
    base = metrics_from_counts(c["tp"], c["fp"], c["fn"])

    tc = compute_tolerance_counts(pred_mask, gt_mask, tolerance_px=tolerance_px)
    tol = tolerance_metrics_from_counts(
        tc["tol_tp_pred"], tc["pred_total"], tc["tol_tp_gt"], tc["gt_total"]
    )
    return {**base, **tol}


@dataclass
class SegmentationMeter:
    """Dataset-level accumulator for global (micro) and image-averaged (macro) metrics."""

    tolerance_px: int = 2
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tol_tp_pred: int = 0
    pred_total: int = 0
    tol_tp_gt: int = 0
    gt_total: int = 0
    num_images: int = 0

    def update(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> None:
        c = compute_confusion_counts(pred_mask, gt_mask)
        self.tp += c["tp"]
        self.fp += c["fp"]
        self.fn += c["fn"]

        tc = compute_tolerance_counts(pred_mask, gt_mask, tolerance_px=self.tolerance_px)
        self.tol_tp_pred += tc["tol_tp_pred"]
        self.pred_total += tc["pred_total"]
        self.tol_tp_gt += tc["tol_tp_gt"]
        self.gt_total += tc["gt_total"]
        self.num_images += 1

    def compute(self) -> Dict[str, float]:
        base = metrics_from_counts(self.tp, self.fp, self.fn)
        tol = tolerance_metrics_from_counts(
            self.tol_tp_pred, self.pred_total, self.tol_tp_gt, self.gt_total
        )
        return {**base, **tol}


def compute_measurement_errors(
    pred_values: Union[Sequence[float], np.ndarray],
    gt_values: Union[Sequence[float], np.ndarray],
) -> Dict[str, float]:
    """Compute MAE, RMSE, and MAPE (%) between predicted and reference measurements.

    MAPE is computed across samples where the reference ground-truth value > 0.
    """
    pred = np.asarray(pred_values, dtype=np.float64)
    gt = np.asarray(gt_values, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    if pred.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape_pct": 0.0}

    abs_err = np.abs(pred - gt)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))

    valid = gt > EPSILON
    if np.any(valid):
        mape_pct = float(np.mean(abs_err[valid] / gt[valid]) * 100.0)
    else:
        mape_pct = 0.0

    return {"mae": mae, "rmse": rmse, "mape_pct": mape_pct}
