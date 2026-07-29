#!/usr/bin/env python3
"""Warm image generation worker. Keeps the diffusion pipeline loaded in one process."""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from zimage.engine import generate_image, load_pipeline
from zimage.paths import get_loras_dir

from .config import apply_env, default_precision, worker_host, worker_port
from .loras import resolve_path

apply_env()
LORAS_DIR = get_loras_dir()


def log(msg: str) -> None:
    print(f"[worker] {msg}", file=sys.stderr, flush=True)


def resolve_loras(entries: list[dict]) -> list[tuple[str, float]]:
    resolved = []
    for entry in entries:
        file = entry["file"]
        strength = float(entry.get("strength", 1.0))
        path = resolve_path(file, LORAS_DIR)
        resolved.append((str(path), strength))
    return resolved


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        log(fmt % args)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok"})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/generate":
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        prompt = body["prompt"]
        output = Path(body["output"])
        width = int(body.get("width", 1024))
        height = int(body.get("height", 1024))
        seed = int(body.get("seed", 42))
        precision = body.get("precision", default_precision())
        steps = int(body.get("steps", 9))
        loras = resolve_loras(body.get("loras", []))

        output.parent.mkdir(parents=True, exist_ok=True)
        image = generate_image(
            prompt=prompt,
            steps=steps,
            width=width,
            height=height,
            seed=seed,
            precision=precision,
            loras=loras or None,
        )
        image.save(output)
        self._json(200, {"output": str(output), "precision": precision, "loras": len(loras)})

    def _json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    precision = sys.argv[1] if len(sys.argv) > 1 else default_precision()
    log(f"warming up precision={precision}")
    load_pipeline(precision=precision)
    log(f"ready on http://{worker_host()}:{worker_port()}")
    HTTPServer((worker_host(), worker_port()), Handler).serve_forever()


if __name__ == "__main__":
    main()
