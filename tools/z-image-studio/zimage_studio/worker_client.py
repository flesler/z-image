from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from .config import VENV_BIN, ROOT, apply_env, worker_host, worker_port
from .loras import LoraSpec, resolve_path
from zimage.engine import generate_image


def health_url() -> str:
    return f"http://{worker_host()}:{worker_port()}/health"


def generate_url() -> str:
    return f"http://{worker_host()}:{worker_port()}/generate"


def generate_batch_url() -> str:
    return f"http://{worker_host()}:{worker_port()}/generate_batch"


def is_healthy() -> bool:
    try:
        with urllib.request.urlopen(health_url(), timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def ensure_worker(cold: bool = False) -> None:
    if cold or is_healthy():
        return
    print("starting warm worker...", flush=True)
    python = VENV_BIN / "python"
    cli = ROOT / "cli.py"
    subprocess.run(
        [str(python), str(cli), "daemon", "start"],
        cwd=ROOT,
        check=True,
    )


def generate_via_worker(
    *,
    prompt: str,
    output: Path,
    width: int,
    height: int,
    seed: int,
    steps: int,
    precision: str,
    loras: list[LoraSpec],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt": prompt,
        "output": str(output),
        "width": width,
        "height": height,
        "seed": seed,
        "steps": steps,
        "precision": precision,
        "loras": [{"file": spec.file, "strength": spec.strength} for spec in loras],
    }
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
    steps: int,
    precision: str,
) -> dict:
    payload = {
        "jobs": jobs,
        "width": width,
        "height": height,
        "steps": steps,
        "precision": precision,
    }
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
        return json.loads(resp.read().decode())


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
    output.parent.mkdir(parents=True, exist_ok=True)
    lora_paths = [(str(resolve_path(spec.file)), spec.strength) for spec in loras]
    image = generate_image(
        prompt=prompt,
        steps=steps,
        width=width,
        height=height,
        seed=seed,
        precision=precision,
        loras=lora_paths or None,
    )
    image.save(output)
