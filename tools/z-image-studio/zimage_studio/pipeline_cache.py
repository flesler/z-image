"""Pipeline load/unload and idle-timeout eviction."""
from __future__ import annotations

import gc
import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterator

import torch
import zimage.engine as zengine

from .config import idle_unload_minutes
from .text_encoder import release_text_encoder


def current_pipe():
    return getattr(zengine, "_cached_pipe", None)


def unload_pipeline() -> bool:
    pipe = current_pipe()
    if pipe is None:
        return False

    release_text_encoder(pipe)

    original = getattr(zengine, "_cached_original_transformer", None)
    zengine._cached_pipe = None
    zengine._cached_precision = None
    zengine._cached_original_transformer = None
    zengine._is_using_compiled_transformer = False

    del original
    del pipe
    for _ in range(2):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return True


def _log(msg: str) -> None:
    print(f"[worker] {msg}", file=sys.stderr, flush=True)


class IdleGuard:
    def __init__(self, timeout_minutes: float | None = None) -> None:
        minutes = idle_unload_minutes() if timeout_minutes is None else timeout_minutes
        self.timeout_s = max(0.0, minutes * 60)
        self._last_activity = time.monotonic()
        self._busy = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self.timeout_s > 0

    def touch(self) -> None:
        with self._lock:
            self._last_activity = time.monotonic()

    @contextmanager
    def job(self) -> Iterator[None]:
        with self._lock:
            self._busy += 1
            self._last_activity = time.monotonic()
        try:
            yield
        finally:
            with self._lock:
                self._busy -= 1
                self._last_activity = time.monotonic()

    def idle_seconds(self) -> float:
        with self._lock:
            return time.monotonic() - self._last_activity

    def start(self) -> None:
        if not self.enabled:
            _log("idle unload disabled (ZIMAGE_IDLE_UNLOAD_MINUTES=0)")
            return
        self._thread = threading.Thread(target=self._loop, name="idle-guard", daemon=True)
        self._thread.start()
        _log(f"idle unload after {self.timeout_s / 60:.0f} min with no generation jobs")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        poll_s = min(30.0, max(5.0, self.timeout_s / 10))
        while not self._stop.wait(poll_s):
            with self._lock:
                if self._busy > 0:
                    continue
                idle_s = time.monotonic() - self._last_activity
            if idle_s < self.timeout_s or current_pipe() is None:
                continue
            if unload_pipeline():
                _log(f"unloaded pipeline after {idle_s / 60:.1f} min idle")
