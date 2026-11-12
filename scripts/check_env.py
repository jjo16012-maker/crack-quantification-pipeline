"""Environment and GPU verification script."""

import sys
import platform


def main() -> None:
    print("=" * 60)
    print("Crack Detection & Quantification - Environment Check")
    print("=" * 60)
    print(f"OS            : {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version.split()[0]}")

    try:
        import torch
        import torchvision

        print(f"PyTorch       : {torch.__version__}")
        print(f"Torchvision   : {torchvision.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")

        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU Device    : {device_name} ({vram_gb:.1f} GB VRAM, count={device_count})")

            # Quick tensor operation on GPU
            x = torch.randn(1024, 1024, device="cuda")
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            print(f"GPU Tensor Op : SUCCESS (output shape={tuple(y.shape)})")
        else:
            print("WARNING: CUDA is not available. Check PyTorch CUDA installation.")
    except ImportError:
        print("PyTorch       : NOT INSTALLED")

    print("-" * 60)
    packages = [
        ("cv2", "opencv-python"),
        ("skimage", "scikit-image"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("yaml", "pyyaml"),
        ("segmentation_models_pytorch", "segmentation-models-pytorch"),
        ("gradio", "gradio"),
    ]
    for module_name, pkg_name in packages:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "installed")
            print(f"{pkg_name:<30}: {version}")
        except ImportError:
            print(f"{pkg_name:<30}: NOT INSTALLED")

    print("=" * 60)


if __name__ == "__main__":
    main()
