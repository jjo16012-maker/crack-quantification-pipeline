"""Unified dataset loader for CUBIT-Seg, DeepCrack, and other crack segmentation datasets.

This module automatically discovers (image, mask) pairs across varying directory
structures and normalizes all ground-truth annotations into clean 2D binary masks
where 0 = background and 1 = crack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 클래스 자동 초기화(__init__)
@dataclass(frozen=True)   # 불변(immutable) 객체로 선언 
class SamplePair:
    # manage data sample(image_path, mask_path, subset name) as pair

    image_path: Path
    mask_path: Path
    subset: str = "default"

    # 서브셋/파일명 형태의 고유 ID 추출 Export ID (subset/filename)
    @property  # 클래스 내 함수를 변수처럼 사용 가능하게 함 (pair.sample_id 형태)
    def sample_id(self) -> str:
        return f"{self.subset}/{self.image_path.stem}"   


def _collect_files_by_stem(directory: Path) -> dict[str, Path]:
    # Map file stem to full path inside a directory. (key/stem, value/path)
    # *stem = 확장자를 제외한 파일명
    mapping: dict[str, Path] = {}
    if not directory.exists():
        return mapping
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            mapping[p.stem] = p
    return mapping


def discover_cubit_seg_pairs(root_dir: Union[str, Path]) -> List[SamplePair]:
    """Discover all (outputs_RGB, outputs_Mask) pairs under CUBIT-Seg root directory.

    Supports nested extraction paths such as:
    data/cubit_seg/.../crack512/{V7,labelme}/{outputs_RGB,outputs_Mask}
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"CUBIT-Seg directory not found: {root}")

    pairs: List[SamplePair] = []
    # *relob: Recursive glob - 재귀적으로 특정 패턴 가진 파일/폴더 탐색하는 기능
    rgb_dirs = sorted(root.rglob("outputs_RGB"))  # 파일명 하드코딩
    # 폴더 내 모든 이미지 폴더 찾음

    for rgb_dir in rgb_dirs:
        # 해당 이미지 폴더의 부모 폴더로 올라감
        parent = rgb_dir.parent
        # 부모 폴더에 마스크 폴더 있는지 확인
        mask_dir = parent / "outputs_Mask"
        if not mask_dir.exists():
            continue

        # 변수에 부모 파일 이름 할당 (데이터 구분 태그로 사용)
        subset_name = parent.name  # e.g., "V7" or "labelme"
        # _collect_files_by_stems: 폴더 내 stem만 정리해 딕셔너리로 만드는 보조 함수
        rgb_map = _collect_files_by_stem(rgb_dir)
        mask_map = _collect_files_by_stem(mask_dir)

        # common_stems: 이미지, 마스크 각 폴더에 공통으로 존재하는 stem list
        # pair 없이 한쪽 폴더에만 있는 파일을 걸러냄. 
        common_stems = sorted(set(rgb_map.keys()) & set(mask_map.keys()))  # 딕셔너리에서 key만 뽑아옴
        for stem in common_stems: 
            pairs.append(
                SamplePair(
                    image_path=rgb_map[stem],
                    mask_path=mask_map[stem],
                    subset=subset_name,
                )
            )

    return pairs  # 반환 형태: List[SamplePair]


def discover_deepcrack_pairs(
    root_dir: Union[str, Path],
    split: str = "all",
) -> List[SamplePair]:
    """Discover (image, mask) pairs under DeepCrack root directory.

    Args:
        root_dir: Root directory containing train_img/train_lab/test_img/test_lab.
        split: One of 'train', 'test', or 'all'.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"DeepCrack directory not found: {root}")

    if split not in {"train", "test", "all"}:
        raise ValueError(f"Invalid split '{split}'. Choose from 'train', 'test', 'all'.")

    target_splits = ["train", "test"] if split == "all" else [split]
    pairs: List[SamplePair] = []

    for sp in target_splits:
        img_dirs = sorted(root.rglob(f"{sp}_img"))
        for img_dir in img_dirs:
            lab_dir = img_dir.parent / f"{sp}_lab"
            if not lab_dir.exists():
                continue

            img_map = _collect_files_by_stem(img_dir)
            lab_map = _collect_files_by_stem(lab_dir)
            common_stems = sorted(set(img_map.keys()) & set(lab_map.keys()))

            for stem in common_stems:
                pairs.append(
                    SamplePair(
                        image_path=img_map[stem],
                        mask_path=lab_map[stem],
                        subset=sp,
                    )
                )

    return pairs


def imread_unicode(path: Union[str, Path], flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    #Read image safely on Windows paths containing Korean/Unicode characters.
    path_obj = Path(path)
    raw_bytes = np.fromfile(str(path_obj), dtype=np.uint8)  # 바이너리 형태로 메모리에 파일 업로드
    img = cv2.imdecode(raw_bytes, flags)  # 바이너리 파일 해독
    if img is None:
        raise IOError(f"Failed to decode image file: {path_obj}")
    return img


def load_rgb_image(path: Union[str, Path]) -> np.ndarray:
    # Load an image as RGB uint8 array of shape (H, W, 3).
    bgr = imread_unicode(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # BGR -> RGB


def load_binary_mask(path: Union[str, Path]) -> np.ndarray:
    """Load any ground-truth mask (palette RGB or grayscale) as binary uint8 (0 or 1).

    In both CUBIT-Seg (palette RGB where crack=[128,0,0]) and DeepCrack (grayscale 0/255),
    background pixels are strictly 0 and foreground crack pixels are > 0.
    """
    raw = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if raw.ndim == 3:
        # Any non-zero channel indicates a foreground crack annotation
        binary = np.any(raw > 0, axis=-1).astype(np.uint8)
    else:
        binary = (raw > 0).astype(np.uint8)
    return binary


class CrackDataset(Dataset):
    # Pytorch의 torch.utils.data.Dataset 상속받은 클래스
    # PyTorch Dataset for crack segmentation and geometric quantification."""

    def __init__(
        self,
        root_dir: Union[str, Path],  # 데이터셋 저장된 root 경로
        dataset_type: str = "auto",  # 폴더 구조를 보고 CUBIT인지 DC인지 추론
        split: str = "all",  # train, test, all 중 어떤 데이터 가져올 지 결정
        resize_to: Optional[Tuple[int, int]] = None,  # 이미지 일괄 리사이징 할 크기
        transform: Optional[Callable] = None,  # Data Augmentation 
        return_tensors: bool = False,  # True->파이토치 텐서로 변환 / False->이미지(Numpy)로 변환
    ) -> None:
        """
        Args:
            root_dir: Path to dataset directory (e.g. 'data/cubit_seg' or 'data/deepcrack').
            dataset_type: 'cubit_seg', 'deepcrack', or 'auto' (inferred from path/structure).
            split: 'all', 'train', or 'test' (used for datasets with predefined splits).
            resize_to: Optional (height, width) tuple to resize image and mask.
            transform: Optional Albumentations-style transform taking (image=..., mask=...).
            return_tensors: If True, converts image to float32 CHW [0,1] tensor and mask to float32 [1,H,W].
        """
        self.root_dir = Path(root_dir)
        self.dataset_type = dataset_type.lower()
        self.split = split
        self.resize_to = resize_to
        self.transform = transform
        self.return_tensors = return_tensors

        # 각 이름에 맞는 폴더를 보고 CUBIT 또는 DC로 종류 변경
        if self.dataset_type == "auto":
            if list(self.root_dir.rglob("outputs_RGB")):
                self.dataset_type = "cubit_seg"
            elif list(self.root_dir.rglob("train_img")) or list(self.root_dir.rglob("test_img")):
                self.dataset_type = "deepcrack"
            else:
                raise ValueError(f"Could not auto-detect dataset structure in {self.root_dir}")

        # 짝을 맞춰 반환한 리스트를 self.pair에 저장
        if self.dataset_type == "cubit_seg":
            self.pairs = discover_cubit_seg_pairs(self.root_dir)
        elif self.dataset_type == "deepcrack":
            self.pairs = discover_deepcrack_pairs(self.root_dir, split=self.split)
        else:
            raise ValueError(f"Unsupported dataset_type: {self.dataset_type}")

        # pair가 안나오면 에러
        if len(self.pairs) == 0:
            raise RuntimeError(f"No (image, mask) pairs found in {self.root_dir}")

    # file pair 개수 확인
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        # 이미지/마스크를 Numpy 배열로 읽어옴
        image = load_rgb_image(pair.image_path)
        mask = load_binary_mask(pair.mask_path)

        # mask.shape: 원본 이미지 크기 불러옴
        orig_h, orig_w = mask.shape[:2]

        # 변수로 resize값 받아오면 작동
        if self.resize_to is not None:
            target_h, target_w = self.resize_to
            if (orig_h, orig_w) != (target_h, target_w):

                # 이미지: 픽셀을 부드럽게 섞어 자연스럽게 하는 선형 보간
                image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                # 마스크: 균열을 선명하게 보기 위해 nearest 보간 
                mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # 변수로 받아오면 데이터 증강
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # return_tensors가 true일 때 텐서로 변환. 모델 훈련 시 True 명시해 넘기면 변환됨. (학습준비)
        if self.return_tensors: # 현재 Flase로 선언
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
            return {
                "image": img_tensor,
                "mask": mask_tensor,
                "sample_id": pair.sample_id,
                "subset": pair.subset,
                "orig_size": (orig_h, orig_w),
                "image_path": str(pair.image_path),
                "mask_path": str(pair.mask_path),
            }

        return {
            "image": image,          # uint8 RGB (H, W, 3)
            "mask": mask,            # uint8 binary {0, 1} (H, W)
            "sample_id": pair.sample_id,
            "subset": pair.subset,
            "orig_size": (orig_h, orig_w),
            "image_path": str(pair.image_path),
            "mask_path": str(pair.mask_path),
        }
