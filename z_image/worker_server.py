#!/usr/bin/env python3
"""Warm image generation worker. Keeps the diffusion pipeline loaded in one process."""
from __future__ import annotations

import json
import mimetypes
import os
import random
import sys
import tempfile
import time
import traceback
import cgi
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from zimage.paths import get_loras_dir

from .config import (
    ROOT,
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
from .gallery import gallery_root, list_images, open_gallery_file
from .metadata import read_prompt
from .naming import output_filename
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
WEB_UI = ROOT / "web" / "index.html"


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
    unload_caption_model()


def run_image_upload(data: bytes, filename: str) -> dict:
    uploads_dir = gallery_root() / ".uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
        suffix = ".png"
    path = uploads_dir / f"current{suffix}"
    path.write_bytes(data)
    return {"path": str(path.resolve())}


def run_caption_upload(data: bytes, filename: str, *, device: str = "auto") -> dict:
    suffix = Path(filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = Path(tmp.name)
    try:
        prompt = read_prompt(path)
        if prompt:
            return {"caption": prompt, "source": "metadata"}
        pipeline_loaded = current_pipe() is not None
        caption = caption_image(path, device=device, pipeline_loaded=pipeline_loaded)
        return {"caption": caption, "source": "blip", "device": caption_model_device()}
    finally:
        path.unlink(missing_ok=True)


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


def _seed_from_filename(name: str) -> int | None:
    import re

    m = re.search(r"-(\d+)(?:-s\d+)?\.png$", name, re.I)
    return int(m.group(1)) if m else None


def run_generate_job(body: dict) -> dict:
    prompt = body["prompt"]
    width = int(body.get("width", 1024))
    height = int(body.get("height", 1024))
    seed = body.get("seed")
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    else:
        seed = int(seed)
    body["seed"] = seed
    steps = resolve_steps(body.get("steps"))
    strength = body.get("strength")
    img_strength = float(strength) if strength is not None else None

    if "output" not in body:
        filename = output_filename(
            prompt,
            width,
            height,
            seed,
            steps,
            strength=img_strength if body.get("image") else None,
        )
        body["output"] = str(gallery_root() / filename)

    output = Path(body["output"])
    precision = resolve_precision(body.get("precision"))
    loras = resolve_loras(body.get("loras", []))

    if output.is_file():
        rel = output.resolve().relative_to(gallery_root())
        existing_seed = _seed_from_filename(output.name) or seed
        log(f"skip existing → {output}")
        return {
            "output": str(output),
            "gallery_path": rel.as_posix(),
            "precision": precision,
            "loras": len(loras),
            "elapsed_s": 0,
            "seed": existing_seed,
            "skipped": True,
        }

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
    rel = output.resolve().relative_to(gallery_root())
    return {
        "output": str(output),
        "gallery_path": rel.as_posix(),
        "precision": precision,
        "loras": len(loras),
        "elapsed_s": round(elapsed, 2),
        "seed": seed,
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
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            if not WEB_UI.is_file():
                self._json(404, {"error": "web ui not found"})
                return
            self._send_file(WEB_UI, content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/health":
            self._json(
                200,
                {
                    "status": "ok",
                    "model_loaded": current_pipe() is not None,
                    "caption_loaded": caption_model_device() is not None,
                    "caption_device": caption_model_device(),
                    "idle_unload_min": idle_unload_minutes(),
                    "idle_s": round(_idle_guard.idle_seconds(), 1),
                    "gallery_root": str(gallery_root()),
                    "gpu": snapshot(current_pipe()),
                },
            )
            return
        if parsed.path == "/gallery":
            params = parse_qs(parsed.query)
            subfolder = params.get("subfolder", [None])[0]
            recursive = params.get("recursive", ["0"])[0] in ("1", "true", "yes")
            try:
                images = list_images(subfolder, recursive=recursive)
            except ValueError as e:
                self._json(400, {"error": str(e)})
                return
            self._json(
                200,
                {
                    "root": str(gallery_root()),
                    "subfolder": subfolder,
                    "images": images,
                },
            )
            return
        if parsed.path == "/gallery/file":
            params = parse_qs(parsed.query)
            rel_path = params.get("path", [None])[0]
            if not rel_path:
                self._json(400, {"error": "path required"})
                return
            try:
                self._send_file(open_gallery_file(rel_path))
            except ValueError as e:
                self._json(400, {"error": str(e)})
            except FileNotFoundError:
                self._json(404, {"error": "not found"})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/caption/upload":
            self._handle_caption_upload()
            return
        if parsed.path == "/upload":
            self._handle_image_upload()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        try:
            with _idle_guard.job():
                if parsed.path == "/caption":
                    self._json(200, run_caption_job(body))
                    return
                if parsed.path == "/generate":
                    self._json(200, run_generate_job(body))
                    return
                if parsed.path == "/generate_batch":
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

    def _handle_image_upload(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._json(400, {"error": "expected multipart/form-data"})
            return
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        if "image" not in form:
            self._json(400, {"error": "image required"})
            return
        field = form["image"]
        if not getattr(field, "file", None):
            self._json(400, {"error": "image required"})
            return
        data = field.file.read()
        filename = field.filename or "upload.png"
        try:
            self._json(200, run_image_upload(data, filename))
        except Exception as e:
            log(f"error: {e}")
            traceback.print_exc()
            self._json(500, {"error": str(e)})

    def _handle_caption_upload(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._json(400, {"error": "expected multipart/form-data"})
            return
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        if "image" not in form:
            self._json(400, {"error": "image required"})
            return
        field = form["image"]
        if not getattr(field, "file", None):
            self._json(400, {"error": "image required"})
            return
        data = field.file.read()
        filename = field.filename or "upload.png"
        device = form.getvalue("device", "auto")
        try:
            with _idle_guard.job():
                result = run_caption_upload(data, filename, device=device)
            self._json(200, result)
        except Exception as e:
            log(f"error: {e}")
            traceback.print_exc()
            self._json(500, {"error": str(e)})

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

    def _send_file(self, path: Path, *, content_type: str | None = None) -> None:
        mime = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(size))
        if mime.startswith("text/html"):
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with path.open("rb") as fh:
            while chunk := fh.read(65536):
                self.wfile.write(chunk)


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
    root = gallery_root()
    log(f"ready on http://{worker_host()}:{worker_port()}")
    log(f"gallery root: {root}")
    HTTPServer((worker_host(), worker_port()), Handler).serve_forever()


if __name__ == "__main__":
    main()
