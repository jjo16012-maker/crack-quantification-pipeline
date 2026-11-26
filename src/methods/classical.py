"""Classical computer vision baseline for crack segmentation using OpenCV.

Pipeline:
1. Grayscale conversion & CLAHE contrast normalization (명암 대비 선명하게 맞춤)
2. Bilateral filtering (suppresses surface grain while preserving crack edges) 
3. Multi-scale morphological Black-Hat transform (highlights dark thin valleys) (국소적인 균열 뽑아냄)
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
        kernel = cv2.getStructuringElement(  # 커널을 (타)원형으로 만들어 방향 편향 없앰
            cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # closing 

    if min_area <= 0 and min_span <= 0:
        return binary


    # num_labels: 전체 라벨 개수 / labels: 각 픽셀 라벨이 담긴 2D 배열 / stats: 라벨별 통계 들어있는 2D 표 / _: 무게중심 (미사용)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return binary

    # [0]은 배경이므로 제외
    areas = stats[1:, cv2.CC_STAT_AREA]  # 객체 픽셀 수(면적)
    widths = stats[1:, cv2.CC_STAT_WIDTH]  # 객체 bounding width
    heights = stats[1:, cv2.CC_STAT_HEIGHT]  # 객체 bounding height
    spans = np.hypot(widths, heights)  # bounding의 대각선 길이 / 원형 노이즈 거르느 용도 (점은 대각선이 짧음)

    valid_ids = np.where((areas >= min_area) & (spans >= min_span))[0] + 1  # +1: 라벨 번호 맞추기
    return np.isin(labels, valid_ids).astype(np.uint8)  # 살아남은 픽셀만 True(1) 반환 (이진 마스크)


def segment_classical(
    image_rgb: np.ndarray,
    clahe_clip: float = 2.0,   # CLAHE에서 대비 증폭 제한값
    clahe_grid: int = 8,       # 대비 맞출 때 이미지 나눌 픽셀 칸 수 (8*8)
    blur_d: int = 5,           # bilateralFilter에서 주변 픽셀 볼 때 창의 지름(픽셀)
    blur_sigma: float = 35.0,  # 밝기 차이가 기준 이내면 비슷하다고 판단해 섞음
    blackhat_kernels: Sequence[int] = (9, 17, 25),  # 어두운 균열 강조할 blackhat 커널 크기 목록
    std_factor: float = 1.8,   # 임계값이 mean + std_factor로 할 때 곱하는 표준편차 배수
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

    # Contrast Limited Adaptive Histogram Equalization: 이미지를 나눈 각 구역의 명암 대비 높임
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid)
        )
        gray = clahe.apply(gray)

    if blur_d > 0:
        # 노이즈 완화 + 경계 유지 (양방향 필터)
        gray = cv2.bilateralFilter(gray, d=blur_d, sigmaColor=blur_sigma, sigmaSpace=blur_sigma)

    # Multi-scale Black-Hat response to capture both thin and wide cracks
    # MORPH_BLACKHAT: (closing 결과) - (원본 이미지) / 주변보다 어두운 구조가 강하게 강조됨 
    response = np.zeros_like(gray, dtype=np.float32)
    for ksize in blackhat_kernels:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
        response = np.maximum(response, bh)  # 균열 두께가 다를 때 각 픽셀별로 스케일에서 나온 결과 중 max값을 할당

    # 이미지마다 밝기/대비가 다르므로 response map 평균 기준으로 계산해 그보다 큰 픽셀만 1로 잡음
    threshold = float(response.mean() + std_factor * response.std())
    raw_mask = (response > threshold).astype(np.uint8)

    return clean_binary_mask(
        raw_mask,
        closing_kernel=closing_kernel,
        min_area=min_area,
        min_span=min_span,
    )
