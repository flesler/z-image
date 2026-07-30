from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_BIN = ROOT / ".venv" / "bin"
LIB_DIR = ROOT / "lib"
LORAS_JSON = Path(os.environ.get("ZIMAGE_LORAS_MAP", LIB_DIR / "loras.json"))
PIDFILE = ROOT / ".worker.pid"
# 225 max filename length, deducts suffix and much to spare
SLUG_MAX = 150
DEFAULT_STEPS = 8
PREVIEW_STEPS = 3
DEFAULT_IMG2IMG_STRENGTH = 0.6

_patch_installed = False


def ensure_gpu_pipeline_patch() -> None:
    global _patch_installed
    if _patch_installed:
        return
    from .gpu_pipeline import install

    install()
    _patch_installed = True


def load_pipeline(*args, **kwargs):
    """Always use the patched zimage.engine.load_pipeline (fast GPU path)."""
    ensure_gpu_pipeline_patch()
    import zimage.engine as zengine

    return zengine.load_pipeline(*args, **kwargs)


def verbose_enabled() -> bool:
    return os.environ.get("ZIMAGE_VERBOSE", "0").lower() in ("1", "true", "yes")


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
    os.environ.setdefault("ZIMAGE_CPU_OFFLOAD", "0")
    if VENV_BIN.is_dir():
        path = os.environ.get("PATH", "")
        venv_path = str(VENV_BIN)
        if venv_path not in path.split(os.pathsep):
            os.environ["PATH"] = f"{venv_path}{os.pathsep}{path}"


def output_dir() -> Path:
    return Path(os.environ.get("ZIMAGE_OUTPUT_DIR", "/tmp/z-image"))


def batch_dir() -> Path:
    return output_dir() / "batch"


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


def resolve_steps(steps: int | None) -> int:
    return DEFAULT_STEPS if steps is None else int(steps)


def resolve_precision(precision: str | None) -> str:
    return default_precision() if precision is None else precision


def resolve_strength(strength: float | None) -> float:
    return DEFAULT_IMG2IMG_STRENGTH if strength is None else float(strength)


def validate_strength(strength: float) -> float:
    if strength < 0.0 or strength > 1.0:
        raise SystemExit("--strength must be between 0.0 and 1.0")
    return strength


def cpu_offload_enabled() -> bool:
    return os.environ.get("ZIMAGE_CPU_OFFLOAD", "0").lower() in ("1", "true", "yes")


def gpu_monitor_enabled() -> bool:
    return os.environ.get("ZIMAGE_GPU_MONITOR", "1").lower() in ("1", "true", "yes")


def idle_unload_minutes() -> float:
    return float(os.environ.get("ZIMAGE_IDLE_UNLOAD_MINUTES", "5"))


def data_dir() -> Path:
    return Path(os.environ.get("Z_IMAGE_STUDIO_DATA_DIR", ROOT / "data"))
