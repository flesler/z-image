from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_BIN = ROOT / ".venv" / "bin"
LIB_DIR = ROOT / "lib"
LORAS_JSON = Path(os.environ.get("ZIMAGE_LORAS_MAP", LIB_DIR / "loras.json"))
PIDFILE = ROOT / ".worker.pid"
SLUG_MAX = 48


def apply_env() -> None:
    root = str(ROOT)
    existing = os.environ.get("PYTHONPATH", "")
    if root not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else root

    data_dir = Path(os.environ.get("Z_IMAGE_STUDIO_DATA_DIR", ROOT / "data"))
    os.environ.setdefault("Z_IMAGE_STUDIO_DATA_DIR", str(data_dir))
    os.environ.setdefault("Z_IMAGE_STUDIO_OUTPUT_DIR", str(data_dir / "outputs"))
    hf_home = data_dir / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if VENV_BIN.is_dir():
        path = os.environ.get("PATH", "")
        venv_path = str(VENV_BIN)
        if venv_path not in path.split(os.pathsep):
            os.environ["PATH"] = f"{venv_path}{os.pathsep}{path}"


def output_dir() -> Path:
    return Path(os.environ.get("ZIMAGE_OUTPUT_DIR", "/tmp/z-image"))


def compare_dir() -> Path:
    return output_dir() / "compare"


def loras_dir() -> Path:
    return Path(os.environ.get("ZIMAGE_LORAS_DIR", ROOT / "data" / "loras"))


def worker_host() -> str:
    return os.environ.get("ZIMAGE_WORKER_HOST", "127.0.0.1")


def worker_port() -> int:
    return int(os.environ.get("ZIMAGE_WORKER_PORT", "18765"))


def worker_log() -> Path:
    return Path(os.environ.get("ZIMAGE_WORKER_LOG", "/tmp/z-image/worker.log"))


def default_precision() -> str:
    return os.environ.get("ZIMAGE_DEFAULT_PRECISION", "q4")
