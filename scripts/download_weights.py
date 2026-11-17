"""Download and verify pretrained weights and dataset integrity (Day 2)."""

from __future__ import annotations

import shutil

import numpy as np
import torch
from huggingface_hub import hf_hub_download
import segmentation_models_pytorch as smp
import timm

from src.data import CrackDataset
from src.paths import DATA_DIR, PROJECT_ROOT, WEIGHTS_DIR

# class CrackDataset 잘 동작하는지 확인
def verify_datasets() -> None:
    print("\n[1/3] Verifying Datasets via src/data.py ...")
    cubit_dir = DATA_DIR / "cubit_seg"
    deepcrack_dir = DATA_DIR / "deepcrack"
    
    cubit_ds = CrackDataset(cubit_dir, dataset_type="cubit_seg")
    v7_count = sum(1 for p in cubit_ds.pairs if p.subset == "V7")
    lm_count = sum(1 for p in cubit_ds.pairs if p.subset == "labelme")
    sample_cubit = cubit_ds[0]
    unique_vals = np.unique(sample_cubit["mask"]).tolist()  # 정답 마스크 구성하는 숫자 출력. [0, 1]만 나와야 함

    print(f"  - CUBIT-Seg Total Pairs : {len(cubit_ds)} (V7={v7_count}, labelme={lm_count})")
    print(
        f"    Sample [0]            : id='{sample_cubit['sample_id']}', "
        f"image={sample_cubit['image'].shape}, mask={sample_cubit['mask'].shape}, "
        f"mask unique values={unique_vals}"
    )

    dc_train = CrackDataset(deepcrack_dir, dataset_type="deepcrack", split="train")
    dc_test = CrackDataset(deepcrack_dir, dataset_type="deepcrack", split="test")
    sample_dc = dc_train[0]
    dc_unique = np.unique(sample_dc["mask"]).tolist()

    print(
        f"  - DeepCrack Total Pairs : {len(dc_train) + len(dc_test)} "
        f"(train={len(dc_train)}, test={len(dc_test)})"
    )
    print(
        f"    Sample [0]            : id='{sample_dc['sample_id']}', "
        f"image={sample_dc['image'].shape}, mask={sample_dc['mask'].shape}, "
        f"mask unique values={dc_unique}"
    )


def verify_and_download_weights() -> None:
    print("\n[2/3] Checking & Downloading Pretrained Crack Model Weights ...")
    weights_dir = WEIGHTS_DIR

    # 1. B1: DeepCrack official checkpoint
    dc_ckpt = weights_dir / "deepcrack" / "DeepCrack_CT260_FT1.pth"
    if dc_ckpt.exists():
        state = torch.load(dc_ckpt, map_location="cpu", weights_only=False)  # 가중치 파일 크기, 내부 구조 출력
        num_keys = len(state) if isinstance(state, dict) else 0
        size_mb = dc_ckpt.stat().st_size / (1024**2)
        print(f"  - [B1] DeepCrack weight found: {dc_ckpt.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB, {num_keys} tensors)")
    else:
        print(f"  - [B1] WARNING: DeepCrack weight not found at {dc_ckpt}")

    # 2. B2: YOLOv8-crack-seg from HuggingFace
    yolo_dir = weights_dir / "yolov8_crack_seg"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    target_yolo_pt = yolo_dir / "yolov8s-crack-seg.pt"

    if not target_yolo_pt.exists():
        print("  - [B2] Downloading YOLOv8s-crack-seg from HuggingFace (OpenSistemas/YOLOv8-crack-seg)...")
        cached_path = hf_hub_download(
            repo_id="OpenSistemas/YOLOv8-crack-seg",
            filename="yolov8s/weights/best.pt",
        )
        shutil.copy2(cached_path, target_yolo_pt)

    yolo_mb = target_yolo_pt.stat().st_size / (1024**2)
    print(f"  - [B2] YOLOv8-crack-seg ready: {target_yolo_pt.relative_to(PROJECT_ROOT)} ({yolo_mb:.1f} MB)")


# 백본 모델 불러오기
def verify_transfer_backbones() -> None:
    print("\n[3/3] Verifying ImageNet Pretrained Backbones for Transfer Learning (D1 & D3) ...")
    # D1 backbone: ConvNeXt-Tiny via timm
    backbone = timm.create_model("convnext_tiny", pretrained=True, features_only=True)
    backbone.eval()
    total_params = sum(p.numel() for p in backbone.parameters()) / 1e6
    print(f"  - [D1 Backbone] timm 'convnext_tiny' loaded ({total_params:.1f}M params)")

    # D3 model: U-Net with ResNet34 encoder via segmentation_models_pytorch
    unet = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    unet.eval()  # 평가 모드로 스위치 전환
    unet_params = sum(p.numel() for p in unet.parameters()) / 1e6
    print(f"  - [D3 Model]    smp.Unet('resnet34', 'imagenet') loaded ({unet_params:.1f}M params)")


def main() -> None:
    print("=" * 70)
    print("Dataset Loader & Pretrained Weights Verification")
    print("=" * 70)
    verify_datasets()
    verify_and_download_weights()
    verify_transfer_backbones()
    print("\ndatasets and pretrained weights are ready!")
    print("=" * 70)


if __name__ == "__main__":
    main()
