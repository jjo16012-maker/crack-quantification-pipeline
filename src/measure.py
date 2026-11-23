"""Geometric quantification of cracks: length and width from a binary mask.

Pipeline per connected crack: skeletonize -> prune short spurs -> sum the
skeleton path length -> read the local width from the distance transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

_EIGHT = np.ones((3, 3), dtype=np.uint8)  # 8-connectivity structuring element


@dataclass(frozen=True)
class CrackMeasurement:
    """Measurements of a single crack, in `pixel_size` units per pixel."""

    length: float
    max_width: float
    mean_width: float


def skeleton_length(skeleton: np.ndarray) -> float:
    """Length of a 1-px-wide skeleton: orthogonal steps count 1, diagonal steps sqrt(2).

    A diagonal link is skipped when the two pixels are already joined through
    an orthogonal path (staircase corners), so corners are not double counted.
    """
    s = skeleton.astype(bool)
    orthogonal = (s[:, :-1] & s[:, 1:]).sum() + (s[:-1, :] & s[1:, :]).sum()
    diagonal = (s[:-1, :-1] & s[1:, 1:] & ~s[:-1, 1:] & ~s[1:, :-1]).sum() + (
        s[:-1, 1:] & s[1:, :-1] & ~s[:-1, :-1] & ~s[1:, 1:]
    ).sum()
    return float(orthogonal + np.sqrt(2.0) * diagonal)


def _prune_spurs(skeleton: np.ndarray, max_spur: int) -> np.ndarray:
    """Remove side branches shorter than `max_spur` pixels that end freely at a junction."""
    if max_spur <= 0:
        return skeleton

    s = skeleton.astype(np.uint8)
    neighbors = ndi.convolve(s, _EIGHT, mode="constant") - s
    endpoints = skeleton & (neighbors == 1)
    junction_zone = ndi.binary_dilation(skeleton & (neighbors >= 3), structure=_EIGHT)

    # Cut the skeleton at junctions so every remaining piece is one branch
    branches, _ = ndi.label(skeleton & ~junction_zone, structure=_EIGHT)
    touches_junction = ndi.binary_dilation(junction_zone, structure=_EIGHT)
    sizes = np.bincount(branches.ravel())

    spur_ids = np.intersect1d(branches[endpoints], branches[touches_junction])
    spur_ids = spur_ids[spur_ids > 0]
    spur_ids = spur_ids[sizes[spur_ids] < max_spur]
    return skeleton & ~np.isin(branches, spur_ids)


def measure_cracks(
    mask: np.ndarray,
    min_area: int = 0,
    max_spur: int = 0,
    pixel_size: float = 1.0,
) -> List[CrackMeasurement]:
    """Measure every connected crack in a binary mask.

    Args:
        mask: 2D array, non-zero pixels are crack.
        min_area: ignore connected components with fewer pixels (noise filter).
        max_spur: prune skeleton side branches shorter than this many pixels.
        pixel_size: physical size of one pixel (e.g. mm/px); 1.0 keeps pixel units.

    Width at a skeleton pixel is `2 * d - 1`, where `d` is its distance to the
    nearest background pixel, which is exact for odd pixel widths.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {binary.shape}")

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    labels, _ = ndi.label(binary, structure=_EIGHT)

    cracks: List[CrackMeasurement] = []
    for label, box in enumerate(ndi.find_objects(labels), start=1):
        component = labels[box] == label
        if component.sum() < min_area:
            continue

        skeleton = _prune_spurs(skeletonize(component), max_spur)
        widths = 2.0 * distance[box][skeleton] - 1.0
        cracks.append(
            CrackMeasurement(
                length=skeleton_length(skeleton) * pixel_size,
                max_width=float(widths.max()) * pixel_size,
                mean_width=float(widths.mean()) * pixel_size,
            )
        )
    return cracks
