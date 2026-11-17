"""Project path constants"""

import os
from pathlib import Path

def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "setup.py").exists() or (p / ".git").exists():
            return p

    raise RuntimeError("프로젝트 루트를 찾지 못했습니다.")

PROJECT_ROOT = Path(os.environ.get("CRACK_ROOT") or _find_project_root())
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RESULTS_DIR = PROJECT_ROOT / "results"
ASSETS_DIR = PROJECT_ROOT / "assets"