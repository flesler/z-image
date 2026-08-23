"""Ensure Hugging Face model weights are present before load."""
from __future__ import annotations

import os
import threading

from huggingface_hub import snapshot_download, try_to_load_from_cache
from zimage.hardware import MODEL_ID_MAP, normalize_precision

from .config import apply_env, resolve_precision
from .log import log

_MARKER = "model_index.json"
_lock = threading.Lock()
_ensured: set[str] = set()


def model_repo_id(precision: str | None = None) -> str:
    key = normalize_precision(resolve_precision(precision))
    return MODEL_ID_MAP[key]


def is_model_cached(precision: str | None = None) -> bool:
    apply_env()
    return try_to_load_from_cache(model_repo_id(precision), _MARKER) is not None


def ensure_model_downloaded(precision: str | None = None) -> str:
    apply_env()
    repo_id = model_repo_id(precision)
    with _lock:
        if repo_id in _ensured:
            return repo_id
        if try_to_load_from_cache(repo_id, _MARKER) is not None:
            _ensured.add(repo_id)
            return repo_id

    log(f"downloading {repo_id} — first run can take several minutes")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    snapshot_download(repo_id, token=token or None)
    with _lock:
        _ensured.add(repo_id)
    log(f"model cached: {repo_id}")
    return repo_id
