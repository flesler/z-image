#!/usr/bin/env python3
"""User-facing entry: venv re-exec + PYTHONPATH bootstrap. Commands live in z_image.app (z-image)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "bin" / "python"


def _ensure_venv() -> None:
    if not VENV_PY.is_file():
        return
    if Path(sys.executable).resolve() == VENV_PY.resolve():
        return
    os.execv(str(VENV_PY), [str(VENV_PY), str(__file__), *sys.argv[1:]])


def _bootstrap() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    existing = os.environ.get("PYTHONPATH", "")
    if root not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else root
    venv_bin = str(ROOT / ".venv" / "bin")
    path = os.environ.get("PATH", "")
    if venv_bin not in path.split(os.pathsep):
        os.environ["PATH"] = f"{venv_bin}{os.pathsep}{path}"


if __name__ == "__main__":
    _ensure_venv()
    _bootstrap()
    t0 = time.perf_counter()
    from z_image.app import main

    raise SystemExit(main(t0=t0))
