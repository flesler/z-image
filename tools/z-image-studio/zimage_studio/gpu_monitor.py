"""GPU/device monitoring for the warm worker."""
from __future__ import annotations

import sys
from typing import Any

import torch

from .config import cpu_offload_enabled


def _module_device(module) -> str | None:
    if module is None:
        return "released"
    try:
        return str(next(module.parameters()).device)
    except StopIteration:
        return "empty"


def _hook_count(pipe) -> int:
    hooks = getattr(pipe, "_all_hooks", None)
    if hooks:
        return len(hooks)
    hooks = getattr(pipe, "_hooks", None)
    if hooks:
        return len(hooks)
    return 0


def snapshot(pipe) -> dict[str, Any]:
    vram_alloc = vram_reserved = vram_peak = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_alloc = torch.cuda.memory_allocated() / 1024**3
        vram_reserved = torch.cuda.memory_reserved() / 1024**3
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3

    devices = {}
    if pipe is not None:
        for name in ("text_encoder", "transformer", "vae"):
            devices[name] = _module_device(getattr(pipe, name, None))

    return {
        "mode": "accelerate-offload" if cpu_offload_enabled() else "fast-gpu",
        "accelerate_hooks": _hook_count(pipe) if pipe is not None else 0,
        "hf_device_map": getattr(pipe, "hf_device_map", None) if pipe is not None else None,
        "devices": devices,
        "vram_gib": round(vram_alloc, 2) if vram_alloc is not None else None,
        "vram_reserved_gib": round(vram_reserved, 2) if vram_reserved is not None else None,
        "vram_peak_gib": round(vram_peak, 2) if vram_peak is not None else None,
    }


def reset_vram_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def format_snapshot(s: dict[str, Any]) -> str:
    dev = s.get("devices") or {}
    parts = [
        f"mode={s.get('mode')}",
        f"hooks={s.get('accelerate_hooks', 0)}",
        f"te={dev.get('text_encoder', '?')}",
        f"tfm={dev.get('transformer', '?')}",
        f"vae={dev.get('vae', '?')}",
    ]
    if s.get("vram_gib") is not None:
        parts.append(f"vram={s['vram_gib']:.2f}GiB")
    if s.get("vram_peak_gib") is not None:
        parts.append(f"peak={s['vram_peak_gib']:.2f}GiB")
    return " ".join(parts)


def log_pipe(pipe, *, label: str, file=None) -> dict[str, Any]:
    snap = snapshot(pipe)
    print(f"[gpu] {label}: {format_snapshot(snap)}", file=file or sys.stderr, flush=True)
    if snap.get("accelerate_hooks", 0) > 0 and not cpu_offload_enabled():
        print(
            "[gpu] WARNING: accelerate hooks active while fast-gpu mode is enabled",
            file=file or sys.stderr,
            flush=True,
        )
    return snap
