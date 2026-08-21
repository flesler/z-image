#!/usr/bin/env python3
"""Warm image generation worker. Keeps the diffusion pipeline loaded in one process."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from zimage.paths import get_loras_dir

from .config import (
    apply_env,
    default_precision,
    ensure_gpu_pipeline_patch,
    idle_unload_minutes,
    load_pipeline,
    resolve_precision,
    resolve_steps,
    resolve_strength,
    worker_host,
    worker_port,
)
from .caption import caption_image, caption_model_device, unload_caption_model
from .exceptions import ClientDisconnected
from .gpu_monitor import log_pipe, reset_vram_peak, snapshot
from .init_image import load_init_image
from .job_monitor import JobMonitor, log_summary
from .loras import resolve_path
from .pipeline_cache import IdleGuard, current_pipe
from .metadata import GenMeta, save_image
from .pipeline_jobs import generate_one, recover_pipe, run_batch_on_pipe
from .prompt_embed_cache import prune, remove_legacy_layout
from .text_encoder import release_text_encoder

apply_env()
ensure_gpu_pipeline_patch()
LORAS_DIR = get_loras_dir()
_idle_guard = IdleGuard()


def log(msg: str) -> None:
    print(f"[worker] {msg}", file=sys.stderr, flush=True)


def job_monitor_enabled() -> bool:
    return os.environ.get("Z_IMAGE_JOB_MONITOR", "1").lower() in ("1", "true", "yes")


def resolve_loras(entries: list[dict]) -> list[tuple[str, float]]:
    resolved = []
    for entry in entries:
        file = entry["file"]
        strength = float(entry.get("strength", 1.0))
        path = resolve_path(file, LORAS_DIR)
        resolved.append((str(path), strength))
    return resolved


def _release_caption_gpu() -> None:
    if caption_model_device() == "cuda":
        unload_caption_model()


def run_caption_job(body: dict) -> dict:
    image = Path(body["image"])
    device = body.get("device", "auto")
    if not image.is_file():
        raise FileNotFoundError(f"image not found: {image}")

    t0 = time.perf_counter()
    pipeline_loaded = current_pipe() is not None
    caption = caption_image(image, device=device, pipeline_loaded=pipeline_loaded)
    elapsed = time.perf_counter() - t0
    log(f"caption in {elapsed:.1f}s ({caption_model_device()}) → {image.name}")
    return {
        "caption": caption,
        "device": caption_model_device(),
        "elapsed_s": round(elapsed, 2),
    }


def run_generate_job(body: dict) -> dict:
    prompt = body["prompt"]
    output = Path(body["output"])
    width = int(body.get("width", 1024))
    height = int(body.get("height", 1024))
    seed = int(body.get("seed", 42))
    precision = resolve_precision(body.get("precision"))
    steps = resolve_steps(body.get("steps"))
    loras = resolve_loras(body.get("loras", []))
    init_image = None
    strength = None
    if body.get("image"):
        strength = resolve_strength(body.get("strength"))
        init_image = load_init_image(body["image"], width=width, height=height)

    output.parent.mkdir(parents=True, exist_ok=True)
    _release_caption_gpu()
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
            init_image=init_image,
            strength=strength,
        )
    except Exception:
        recover_pipe(current_pipe())
        raise
    finally:
        if monitor_ctx:
            monitor_ctx.__exit__(None, None, None)
    elapsed = time.perf_counter() - t0
    save_image(
        image,
        output,
        GenMeta(
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            precision=precision,
            loras=loras or None,
            strength=strength,
        ),
    )
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


def run_generate_batch(body: dict, *, on_image=None) -> dict:
    jobs = body["jobs"]
    if not jobs:
        raise ValueError("jobs must not be empty")

    width = int(body.get("width", 1024))
    height = int(body.get("height", 1024))
    precision = resolve_precision(body.get("precision"))
    init_image = None
    strength = None
    if body.get("image"):
        strength = resolve_strength(body.get("strength"))
        init_image = load_init_image(body["image"], width=width, height=height)

    reset_vram_peak()
    _release_caption_gpu()
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
            reuse_steps=body.get("reuse_steps", False),
            init_image=init_image,
            strength=strength,
            log=log,
            reloaded=reloaded,
            on_image=on_image,
        )
    except ClientDisconnected:
        log("batch cancelled: client disconnected")
        raise
    finally:
        recover_pipe(pipe)
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
                    "caption_loaded": caption_model_device() is not None,
                    "caption_device": caption_model_device(),
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
                if self.path == "/caption":
                    self._json(200, run_caption_job(body))
                    return
                if self.path == "/generate":
                    self._json(200, run_generate_job(body))
                    return
                if self.path == "/generate_batch":
                    if body.get("stream"):
                        self._stream_batch(body)
                    else:
                        self._json(200, run_generate_batch(body))
                    return
        except ClientDisconnected:
            log("client disconnected")
            return
        except Exception as e:
            log(f"error: {e}")
            traceback.print_exc()
            self._json(500, {"error": str(e)})
            return

        self._json(404, {"error": "not found"})

    def _stream_batch(self, body: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()

        disconnected = False

        def write_event(obj: dict) -> None:
            nonlocal disconnected
            try:
                self.wfile.write((json.dumps(obj) + "\n").encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError) as e:
                disconnected = True
                raise ClientDisconnected("client disconnected") from e

        try:
            def on_image(result: dict) -> None:
                write_event({"type": "image", **result})

            summary = run_generate_batch(body, on_image=on_image)
            if not disconnected:
                write_event({"type": "done", **summary})
        except ClientDisconnected:
            log("client disconnected during batch stream")
        except Exception as e:
            log(f"error: {e}")
            traceback.print_exc()
            if not disconnected:
                try:
                    write_event({"type": "error", "error": str(e)})
                except ClientDisconnected:
                    log("client disconnected during error response")
        finally:
            recover_pipe(current_pipe())

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
    load_pipeline(precision=precision)
    release_text_encoder(current_pipe())
    log_pipe(current_pipe(), label="warmup")
    _idle_guard.start()
    log(f"ready on http://{worker_host()}:{worker_port()}")
    HTTPServer((worker_host(), worker_port()), Handler).serve_forever()


if __name__ == "__main__":
    main()
