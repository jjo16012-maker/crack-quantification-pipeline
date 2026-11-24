"""Unified benchmark evaluation script for crack segmentation and quantification.

Evaluates a specified segmentation method on the target dataset (default: CUBIT-Seg),
computes pixel segmentation metrics (F1, Tol-F1, IoU, Precision, Recall) and
geometric quantification errors (Length MAPE, Max/Mean Width MAE), and appends/updates
the summary table in `results/benchmark_summary.csv`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data import CrackDataset, load_binary_mask
from src.measure import CrackMeasurement, measure_cracks
from src.methods.classical import segment_classical
from src.metrics import SegmentationMeter, compute_measurement_errors
from src.paths import DATA_DIR, RESULTS_DIR


def summarize_image_measurements(cracks: List[CrackMeasurement]) -> Dict[str, float]:
    """Aggregate per-component crack measurements into image-level summary values."""
    if not cracks:
        return {"total_length": 0.0, "max_width": 0.0, "mean_width": 0.0}

    total_length = float(sum(c.length for c in cracks))
    max_width = float(max(c.max_width for c in cracks))
    # Length-weighted average width across components in the image
    weights = np.array([max(c.length, 1.0) for c in cracks], dtype=np.float64)
    means = np.array([c.mean_width for c in cracks], dtype=np.float64)
    mean_width = float(np.sum(weights * means) / np.sum(weights))
    return {
        "total_length": total_length,
        "max_width": max_width,
        "mean_width": mean_width,
    }


def load_or_build_gt_measurements(
    dataset: CrackDataset,
    cache_csv: Path,
    min_area: int,
    max_spur: int,
) -> pd.DataFrame:
    """Load cached ground-truth length/width measurements or compute them once."""
    if cache_csv.exists():
        df = pd.read_csv(cache_csv)
        if len(df) == len(dataset):
            return df

    records = []
    for pair in tqdm(dataset.pairs, desc="Computing reference GT measurements"):
        gt_mask = load_binary_mask(pair.mask_path)
        cracks = measure_cracks(gt_mask, min_area=min_area, max_spur=max_spur)
        agg = summarize_image_measurements(cracks)
        records.append(
            {
                "sample_id": pair.sample_id,
                "num_cracks": len(cracks),
                "gt_length_px": agg["total_length"],
                "gt_max_width_px": agg["max_width"],
                "gt_mean_width_px": agg["mean_width"],
            }
        )

    df = pd.DataFrame(records)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    return df


def build_predictor(method: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return an inference callable `fn(image_rgb) -> pred_binary_mask` for a method."""
    key = method.lower()
    if key == "classical":
        return segment_classical
    raise ValueError(f"Unknown or not-yet-registered method: '{method}'")


def evaluate_method(
    method: str,
    dataset_name: str = "cubit_seg",
    split: str = "all",
    tolerance_px: int = 2,
    min_area: int = 20,
    max_spur: int = 5,
    limit: int = 0,
) -> Dict[str, float]:
    """Run full segmentation + geometric measurement benchmark for a method."""
    ds_dir = DATA_DIR / dataset_name
    dataset = CrackDataset(ds_dir, dataset_type=dataset_name, split=split)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gt_cache_path = RESULTS_DIR / f"gt_measurements_{dataset_name}_{split}.csv"

    if limit > 0:
        dataset.pairs = dataset.pairs[:limit]
        gt_df = None
    else:
        gt_df = load_or_build_gt_measurements(
            dataset, gt_cache_path, min_area=min_area, max_spur=max_spur
        )

    predictor = build_predictor(method)
    meter = SegmentationMeter(tolerance_px=tolerance_px)

    pred_lengths, gt_lengths = [], []
    pred_max_widths, gt_max_widths = [], []
    pred_mean_widths, gt_mean_widths = [], []
    latencies_ms = []

    for idx in tqdm(range(len(dataset)), desc=f"Evaluating [{method}] on {dataset_name}"):
        sample = dataset[idx]
        image = sample["image"]
        gt_mask = sample["mask"]

        t0 = time.perf_counter()
        pred_mask = predictor(image)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        meter.update(pred_mask, gt_mask)

        pred_cracks = measure_cracks(pred_mask, min_area=min_area, max_spur=max_spur)
        pred_agg = summarize_image_measurements(pred_cracks)

        if gt_df is not None:
            row = gt_df.iloc[idx]
            gt_len = float(row["gt_length_px"])
            gt_max_w = float(row["gt_max_width_px"])
            gt_mean_w = float(row["gt_mean_width_px"])
        else:
            gt_cracks = measure_cracks(gt_mask, min_area=min_area, max_spur=max_spur)
            gt_agg = summarize_image_measurements(gt_cracks)
            gt_len = gt_agg["total_length"]
            gt_max_w = gt_agg["max_width"]
            gt_mean_w = gt_agg["mean_width"]

        pred_lengths.append(pred_agg["total_length"])
        gt_lengths.append(gt_len)
        pred_max_widths.append(pred_agg["max_width"])
        gt_max_widths.append(gt_max_w)
        pred_mean_widths.append(pred_agg["mean_width"])
        gt_mean_widths.append(gt_mean_w)

    seg_metrics = meter.compute()
    len_err = compute_measurement_errors(pred_lengths, gt_lengths)
    max_w_err = compute_measurement_errors(pred_max_widths, gt_max_widths)
    mean_w_err = compute_measurement_errors(pred_mean_widths, gt_mean_widths)

    result = {
        "method": method,
        "dataset": f"{dataset_name} ({split})",
        "num_images": len(dataset),
        "f1_pct": round(seg_metrics["f1"] * 100.0, 2),
        "tol_f1_pct": round(seg_metrics["tol_f1"] * 100.0, 2),
        "iou_pct": round(seg_metrics["iou"] * 100.0, 2),
        "precision_pct": round(seg_metrics["precision"] * 100.0, 2),
        "recall_pct": round(seg_metrics["recall"] * 100.0, 2),
        "length_mape_pct": round(len_err["mape_pct"], 2),
        "length_mae_px": round(len_err["mae"], 2),
        "max_width_mae_px": round(max_w_err["mae"], 2),
        "mean_width_mae_px": round(mean_w_err["mae"], 2),
        "latency_ms": round(float(np.mean(latencies_ms)), 2),
    }

    if limit == 0:
        summary_csv = RESULTS_DIR / "benchmark_summary.csv"
        if summary_csv.exists():
            df_sum = pd.read_csv(summary_csv)
            mask_existing = (df_sum["method"] == result["method"]) & (
                df_sum["dataset"] == result["dataset"]
            )
            df_sum = df_sum[~mask_existing]
            df_sum = pd.concat([df_sum, pd.DataFrame([result])], ignore_index=True)
        else:
            df_sum = pd.DataFrame([result])
        df_sum.to_csv(summary_csv, index=False)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate crack segmentation & quantification.")
    parser.add_argument("--method", type=str, default="classical", help="Method name")
    parser.add_argument("--dataset", type=str, default="cubit_seg", help="Dataset name")
    parser.add_argument("--split", type=str, default="all", help="Dataset split")
    parser.add_argument("--tolerance-px", type=int, default=2, help="Boundary tolerance (px)")
    parser.add_argument("--min-area", type=int, default=20, help="Min crack area (px)")
    parser.add_argument("--max-spur", type=int, default=5, help="Max spur length to prune (px)")
    parser.add_argument("--limit", type=int, default=0, help="Optional subset limit for quick check")
    args = parser.parse_args()

    res = evaluate_method(
        method=args.method,
        dataset_name=args.dataset,
        split=args.split,
        tolerance_px=args.tolerance_px,
        min_area=args.min_area,
        max_spur=args.max_spur,
        limit=args.limit,
    )

    print("\n" + "=" * 65)
    print(f"Benchmark Result: {res['method']} on {res['dataset']} (n={res['num_images']})")
    print("=" * 65)
    for k, v in res.items():
        if k not in {"method", "dataset", "num_images"}:
            print(f"  {k:<20}: {v}")
    print("=" * 65)


if __name__ == "__main__":
    main()
