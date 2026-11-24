"""Classical computer vision baseline for crack segmentation using OpenCV.

Pipeline:
1. Grayscale conversion & CLAHE contrast normalization
2. Bilateral filtering (suppresses surface grain while preserving crack edges)
3. Multi-scale morphological Black-Hat transform (highlights dark thin valleys)
4. Standard-deviation thresholding -> morphological closing -> connected-component filtering
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def clean_binary_mask(
    mask: np.ndarray,
    closing_kernel: int = 3,
    min_area: int = 40,
    min_span: float = 0.0,
) -> np.ndarray:
    """Apply morphological closing and remove small/compact noise components.

    Args:
        mask: 2D binary array.
        closing_kernel: Elliptical kernel size for connecting fragmented crack segments.
        min_area: Minimum pixel area to keep a connected component.
        min_span: Minimum bounding-box diagonal length (px) to filter out round blobs.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)

    if closing_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if min_area <= 0 and min_span <= 0:
        return binary

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    spans = np.hypot(widths, heights)

    valid_ids = np.where((areas >= min_area) & (spans >= min_span))[0] + 1
    return np.isin(labels, valid_ids).astype(np.uint8)


def segment_classical(
    image_rgb: np.ndarray,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
    blur_d: int = 5,
    blur_sigma: float = 35.0,
    blackhat_kernels: Sequence[int] = (9, 17, 25),
    std_factor: float = 1.8,
    closing_kernel: int = 3,
    min_area: int = 60,
    min_span: float = 18.0,
) -> np.ndarray:
    """Segment dark, elongated crack structures from an RGB image using OpenCV.

    Returns:
        2D binary mask of dtype uint8 with values in {0, 1}.
    """
    if image_rgb.ndim == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb.copy()

    if clahe_clip > 0:
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid)
        )
        gray = clahe.apply(gray)

    if blur_d > 0:
        gray = cv2.bilateralFilter(gray, d=blur_d, sigmaColor=blur_sigma, sigmaSpace=blur_sigma)

    # Multi-scale Black-Hat response to capture both thin and wide cracks
    response = np.zeros_like(gray, dtype=np.float32)
    for ksize in blackhat_kernels:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
        response = np.maximum(response, bh)

    threshold = float(response.mean() + std_factor * response.std())
    raw_mask = (response > threshold).astype(np.uint8)

    return clean_binary_mask(
        raw_mask,
        closing_kernel=closing_kernel,
        min_area=min_area,
        min_span=min_span,
    )
