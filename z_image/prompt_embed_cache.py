"""Disk cache for prompt embeddings — skip text encoder on repeat prompts."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from zimage.hardware import MODEL_ID_MAP

from .config import data_dir
from .loras import normalize_prompt

_last_prune_at = 0.0
PRUNE_INTERVAL_S = 3600


def cache_enabled() -> bool:
    return os.environ.get("Z_IMAGE_PROMPT_EMBED_CACHE", "1").lower() in ("1", "true", "yes")


def cache_root() -> Path:
    path = Path(os.environ.get("Z_IMAGE_PROMPT_EMBED_CACHE_DIR", data_dir() / "cache" / "prompt_embeds"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_id_for(precision: str) -> str:
    return MODEL_ID_MAP[precision]


def cache_hash(prompt: str, precision: str) -> str:
    normalized = normalize_prompt(prompt)
    model_id = model_id_for(precision)
    return hashlib.sha256(f"{model_id}\0{precision}\0{normalized}".encode()).hexdigest()


def _tensor_path(digest: str) -> Path:
    return cache_root() / f"{digest}.pt"


def _meta_path(digest: str) -> Path:
    return cache_root() / f"{digest}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_meta(digest: str) -> dict[str, Any] | None:
    path = _meta_path(digest)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_meta(digest: str, meta: dict[str, Any]) -> None:
    _meta_path(digest).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def load(prompt: str, precision: str) -> list[torch.Tensor] | None:
    if not cache_enabled():
        return None

    digest = cache_hash(prompt, precision)
    tensor_path = _tensor_path(digest)
    if not tensor_path.is_file():
        return None

    meta = _read_meta(digest)
    if meta is None:
        return None
    if meta.get("precision") != precision or meta.get("model_id") != model_id_for(precision):
        return None
    if meta.get("prompt") != normalize_prompt(prompt):
        return None

    data = torch.load(tensor_path, map_location="cpu", weights_only=False)
    embeds = data.get("embeds")
    if not embeds:
        return None

    meta["last_used_at"] = _now_iso()
    meta["hits"] = int(meta.get("hits", 0)) + 1
    _write_meta(digest, meta)
    return embeds


def save(prompt: str, precision: str, embeds: list[torch.Tensor], *, encode_time_s: float) -> str:
    if not cache_enabled():
        return cache_hash(prompt, precision)

    digest = cache_hash(prompt, precision)
    tensor_path = _tensor_path(digest)
    cpu_embeds = [tensor.detach().cpu() for tensor in embeds]
    torch.save({"embeds": cpu_embeds}, tensor_path)

    now = _now_iso()
    meta = {
        "hash": digest,
        "prompt": normalize_prompt(prompt),
        "precision": precision,
        "model_id": model_id_for(precision),
        "created_at": now,
        "last_used_at": now,
        "encode_time_s": round(encode_time_s, 3),
        "hits": 0,
        "size_bytes": tensor_path.stat().st_size,
    }
    _write_meta(digest, meta)
    return digest


def to_device(embeds: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [tensor.to(device) for tensor in embeds]


def list_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for meta_path in sorted(cache_root().glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            digest = meta.get("hash", meta_path.stem)
            pt = _tensor_path(digest)
            meta["present"] = pt.is_file()
            if pt.is_file():
                meta["size_bytes"] = pt.stat().st_size
            entries.append(meta)
        except (json.JSONDecodeError, OSError):
            continue
    return entries


def remove_entry(digest: str) -> None:
    _tensor_path(digest).unlink(missing_ok=True)
    _meta_path(digest).unlink(missing_ok=True)


def remove_legacy_layout() -> int:
    """Drop old per-precision subdirs (q4/, q8/, ...)."""
    removed = 0
    root = cache_root()
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        for path in sub.glob("*.pt"):
            path.unlink(missing_ok=True)
            removed += 1
        for path in sub.glob("*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        try:
            sub.rmdir()
        except OSError:
            pass
    return removed


def maybe_prune() -> dict[str, int] | None:
    """Prune stale entries at most once per hour during active encoding."""
    if not cache_enabled():
        return None
    global _last_prune_at
    now = time.time()
    if now - _last_prune_at < PRUNE_INTERVAL_S:
        return None
    _last_prune_at = now
    return prune()


def prune(
    *,
    max_age_hours: float | None = None,
    max_entries: int | None = None,
    max_mb: int | None = None,
) -> dict[str, int]:
    if max_age_hours is None:
        max_age_hours = float(os.environ.get("Z_IMAGE_PROMPT_EMBED_CACHE_MAX_AGE_HOURS", "24"))
    max_entries = max_entries if max_entries is not None else int(
        os.environ.get("Z_IMAGE_PROMPT_EMBED_CACHE_MAX_ENTRIES", "500")
    )
    max_mb = max_mb if max_mb is not None else int(
        os.environ.get("Z_IMAGE_PROMPT_EMBED_CACHE_MAX_MB", "256")
    )

    remove_legacy_layout()
    entries = [e for e in list_entries() if e.get("present")]
    removed = 0
    now = time.time()

    def age_hours(entry: dict[str, Any]) -> float:
        stamp = entry.get("last_used_at") or entry.get("created_at")
        if not stamp:
            return 999999.0
        try:
            dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            return (now - dt.timestamp()) / 3600
        except ValueError:
            return 999999.0

    for entry in list(entries):
        if age_hours(entry) > max_age_hours:
            remove_entry(entry["hash"])
            entries.remove(entry)
            removed += 1

    def total_mb() -> float:
        return sum(int(e.get("size_bytes", 0)) for e in entries) / (1024 * 1024)

    entries.sort(key=lambda e: e.get("last_used_at") or e.get("created_at") or "")
    while len(entries) > max_entries:
        remove_entry(entries[0]["hash"])
        entries.pop(0)
        removed += 1

    entries = [e for e in list_entries() if e.get("present")]
    entries.sort(key=lambda e: e.get("last_used_at") or e.get("created_at") or "")
    while total_mb() > max_mb and entries:
        remove_entry(entries[0]["hash"])
        entries.pop(0)
        removed += 1

    return {"removed": removed, "remaining": len(list_entries())}
