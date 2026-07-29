#!/usr/bin/env python3
"""Warm image generation worker. Keeps the diffusion pipeline loaded in one process."""
from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import zimage.engine as zengine
from zimage.paths import get_loras_dir

from .config import apply_env, default_precision, idle_unload_minutes, load_pipeline, worker_host, worker_port
from .gpu_monitor import log_pipe, reset_vram_peak, snapshot
from .job_monitor import JobMonitor, log_summary
from .loras import resolve_path
from .pipeline_cache import IdleGuard, current_pipe
from .pipeline_jobs import generate_one, run_batch_on_pipe
from .prompt_embed_cache import prune, remove_legacy_layout
from .text_encoder import release_text_encoder

apply_env()
LORAS_DIR = get_loras_dir()
_idle_guard = IdleGuard()


def log(msg: str) -> None:
    print(f"[worker] {msg}", file=sys.stderr, flush=True)


def job_monitor_enabled() -> bool:
    return os.environ.get("ZIMAGE_JOB_MONITOR", "1").lower() in ("1", "true", "yes")


def resolve_loras(entries: list[dict]) -> list[tuple[str, float]]:
    resolved = []
    for entry in entries:
        file = entry["file"]
        strength = float(entry.get("strength", 1.0))
        path = resolve_path(file, LORAS_DIR)
        resolved.append((str(path), strength))
    return resolved


def run_generate_job(body: dict) -> dict:
    prompt = body["prompt"]
    output = Path(body["output"])
    width = int(body.get("width", 1024))
    height = int(body.get("height", 1024))
    seed = int(body.get("seed", 42))
    precision = body.get("precision", default_precision())
    steps = int(body.get("steps", 9))
    loras = resolve_loras(body.get("loras", []))

    output.parent.mkdir(parents=True, exist_ok=True)
    reset_vram_peak()
    log_pipe(current_pipe(), label="pre-gen")
    monitor_ctx = JobMonitor() if job_monitor_enabled() else None
    t0 = time.perf_counter()
    if monitor_ctx:
        monitor_ctx.__enter__()
    try:
        image = generate_one(
            prompt=prompt,
            steps=steps,
            width=width,
            height=height,
            seed=seed,
            precision=precision,
            loras=loras or None,
        )
    finally:
        if monitor_ctx:
            monitor_ctx.__exit__(None, None, None)
    elapsed = time.perf_counter() - t0
    image.save(output)
    post = log_pipe(current_pipe(), label="post-gen")
    monitor = monitor_ctx.summary() if monitor_ctx else None
    if monitor:
        log_summary(monitor, label="gen")
    log(f"done in {elapsed:.1f}s loras={len(loras)} → {output}")
    return {
        "output": str(output),
        "precision": precision,
        "loras": len(loras),
        "elapsed_s": round(elapsed, 2),
        "gpu": post,
        "monitor": monitor,
    }


def run_generate_batch(body: dict) -> dict:
    jobs = body["jobs"]
    if not jobs:
        raise ValueError("jobs must not be empty")

    width = int(body.get("width", 1024))
    height = int(body.get("height", 1024))
    steps = int(body.get("steps", 9))
    precision = body.get("precision", default_precision())

    reset_vram_peak()
    reloaded = current_pipe() is None
    pipe = load_pipeline(precision=precision)
    monitor_ctx = JobMonitor() if job_monitor_enabled() else None
    if monitor_ctx:
        monitor_ctx.__enter__()

    try:
        resolved_jobs = []
        for job in jobs:
            loras = resolve_loras(job.get("loras", []))
            resolved_jobs.append({**job, "loras": loras})
        t0 = time.perf_counter()
        results, encode_s, denoise_s = run_batch_on_pipe(
            pipe,
            resolved_jobs,
            width=width,
            height=height,
            default_steps=steps,
            log=log,
            reloaded=reloaded,
        )
        release_text_encoder(pipe)
    finally:
        monitor = monitor_ctx.summary() if monitor_ctx else None
        if monitor_ctx:
            monitor_ctx.__exit__(None, None, None)

    total_s = time.perf_counter() - t0
    post = log_pipe(pipe, label="post-batch")
    if monitor:
        log_summary(monitor, label="batch")
    log(f"batch total {total_s:.1f}s = encode {encode_s:.1f}s + denoise {denoise_s:.1f}s ({len(jobs)} images)")
    return {
        "count": len(jobs),
        "prompts_encoded": len({job["prompt"] for job in jobs}),
        "elapsed_s": round(total_s, 2),
        "encode_s": round(encode_s, 2),
        "denoise_s": round(denoise_s, 2),
        "results": results,
        "gpu": post,
        "monitor": monitor,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        log(fmt % args)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(
                200,
                {
                    "status": "ok",
                    "model_loaded": current_pipe() is not None,
                    "idle_unload_min": idle_unload_minutes(),
                    "idle_s": round(_idle_guard.idle_seconds(), 1),
                    "gpu": snapshot(current_pipe()),
                },
            )
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        try:
            with _idle_guard.job():
                if self.path == "/generate":
                    self._json(200, run_generate_job(body))
                    return
                if self.path == "/generate_batch":
                    self._json(200, run_generate_batch(body))
                    return
        except Exception as e:
            log(f"error: {e}")
            self._json(500, {"error": str(e)})
            return

        self._json(404, {"error": "not found"})

    def _json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    precision = sys.argv[1] if len(sys.argv) > 1 else default_precision()
    legacy = remove_legacy_layout()
    if legacy:
        log(f"removed {legacy} legacy cache file(s) from per-precision subdirs")
    pruned = prune()
    if pruned["removed"]:
        log(f"cache prune: removed {pruned['removed']}, {pruned['remaining']} remaining")

    log(f"warming up precision={precision}")
    zengine.load_pipeline(precision=precision)
    release_text_encoder(current_pipe())
    log_pipe(current_pipe(), label="warmup")
    _idle_guard.start()
    log(f"ready on http://{worker_host()}:{worker_port()}")
    HTTPServer((worker_host(), worker_port()), Handler).serve_forever()


if __name__ == "__main__":
    main()
