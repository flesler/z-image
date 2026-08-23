from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import VENV_BIN, ROOT, apply_env, worker_connect_host, worker_port
from .log import log
from .loras import LoraSpec, resolve_path
from .metadata import GenMeta, save_image


def health_url() -> str:
    return f"http://{worker_connect_host()}:{worker_port()}/health"


def caption_url() -> str:
    return f"http://{worker_connect_host()}:{worker_port()}/caption"


def generate_url() -> str:
    return f"http://{worker_connect_host()}:{worker_port()}/generate"


def generate_batch_url() -> str:
    return f"http://{worker_connect_host()}:{worker_port()}/generate_batch"


def fetch_health() -> dict | None:
    try:
        with urllib.request.urlopen(health_url(), timeout=2) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def is_healthy() -> bool:
    return fetch_health() is not None


def ensure_worker() -> None:
    if is_healthy():
        return
    log("starting worker...")
    python = VENV_BIN / "python"
    cli = ROOT / "cli.py"
    subprocess.run(
        [str(python), str(cli), "daemon", "start"],
        cwd=ROOT,
        check=True,
    )
    for _ in range(60):
        if is_healthy():
            return
        time.sleep(0.5)
    raise RuntimeError("worker failed to start within 30s")


def caption_via_worker(path: Path | str, *, device: str = "auto") -> str:
    payload = {"image": str(Path(path).expanduser().resolve()), "device": device}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        caption_url(),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        if resp.status != 200:
            raise RuntimeError(f"worker returned {resp.status}")
        body = json.loads(resp.read())
        return body["caption"]


def generate_via_worker(
    *,
    prompt: str,
    output: Path,
    width: int,
    height: int,
    seed: int,
    steps: int | None = None,
    seed_base: int | None = None,
    precision: str | None = None,
    loras: list[LoraSpec],
    image: Path | None = None,
    strength: float | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "prompt": prompt,
        "output": str(output),
        "width": width,
        "height": height,
        "seed": seed,
        "loras": [{"name": spec.name, "strength": spec.strength} for spec in loras],
    }
    if seed_base is not None:
        payload["seed_base"] = seed_base
    if steps is not None:
        payload["steps"] = steps
    if precision is not None:
        payload["precision"] = precision
    if image is not None:
        payload["image"] = str(image)
        if strength is not None:
            payload["strength"] = strength
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        generate_url(),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        if resp.status != 200:
            raise RuntimeError(f"worker returned {resp.status}")


def generate_batch_via_worker(
    *,
    jobs: list[dict],
    width: int,
    height: int,
    precision: str | None,
    reuse_steps: bool = False,
    image: Path | None = None,
    strength: float | None = None,
    on_image=None,
) -> dict:
    payload: dict = {
        "jobs": jobs,
        "width": width,
        "height": height,
        "reuse_steps": reuse_steps,
        "stream": True,
    }
    if precision is not None:
        payload["precision"] = precision
    if image is not None:
        payload["image"] = str(image)
        if strength is not None:
            payload["strength"] = strength
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        generate_batch_url(),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        if resp.status != 200:
            raise RuntimeError(f"worker returned {resp.status}")
        summary: dict = {}
        for raw in resp:
            line = raw.decode().strip()
            if not line:
                continue
            event = json.loads(line)
            kind = event.get("type")
            if kind == "image":
                if on_image:
                    on_image(event)
            elif kind == "done":
                summary = event
            elif kind == "error":
                raise RuntimeError(event.get("error", "worker batch failed"))
        return summary


def generate_cold(
    *,
    prompt: str,
    output: Path,
    width: int,
    height: int,
    seed: int,
    steps: int,
    precision: str,
    loras: list[LoraSpec],
    extra: list[str] | None = None,
) -> None:
    apply_env()
    from .config import ensure_gpu_pipeline_patch

    ensure_gpu_pipeline_patch()
    from zimage.engine import generate_image

    output.parent.mkdir(parents=True, exist_ok=True)
    lora_paths = [(str(resolve_path(spec.name)), spec.strength) for spec in loras]
    image = generate_image(
        prompt=prompt,
        steps=steps,
        width=width,
        height=height,
        seed=seed,
        precision=precision,
        loras=lora_paths or None,
    )
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
            loras=lora_paths or None,
        ),
    )
