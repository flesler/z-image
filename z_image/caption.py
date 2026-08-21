"""Reverse-engineer natural-language prompts from images (BLIP-large)."""
from __future__ import annotations

import gc
import os
import re
import sys
from pathlib import Path
from typing import Literal

from PIL import Image

from .config import apply_env
from .metadata import embed_prompt as write_prompt_metadata
from .metadata import read_prompt as read_prompt_metadata

DEFAULT_MODEL = "Salesforce/blip-image-captioning-large"
DEFAULT_PROMPT = "a detailed photograph showing"
# BLIP-large fp16 on GPU + beam-search activations + headroom.
GPU_VRAM_REQUIRED_BYTES = int(1.0 * 1024**3)

CaptionDevicePref = Literal["auto", "cpu", "gpu", "cuda"]

_processor = None
_model = None
_model_id: str | None = None
_model_device: str | None = None


def _load_image(path: str | Path) -> Image.Image:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"image not found: {resolved}")
    with Image.open(resolved) as im:
        return im.convert("RGB")


def image_dimensions(path: str | Path) -> tuple[int, int]:
    with Image.open(Path(path).expanduser().resolve()) as im:
        return im.size


def cuda_free_bytes() -> int:
    import torch

    if not torch.cuda.is_available():
        return 0
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def resolve_caption_device(
    preference: CaptionDevicePref | str = "auto",
    *,
    pipeline_loaded: bool = False,
) -> str:
    """Return runtime device: 'cuda' or 'cpu'."""
    import torch

    pref = preference.lower().strip()
    env = os.environ.get("Z_IMAGE_CAPTION_DEVICE", "").lower().strip()
    if pref == "auto" and env:
        pref = env

    if pref in ("cpu",):
        return "cpu"
    if pref in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("--caption-device gpu requested but CUDA is not available")
        return "cuda"
    if pref != "auto":
        raise SystemExit(f"unknown --caption-device {preference!r}; use auto, cpu, or gpu")

    if pipeline_loaded:
        return "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    free = cuda_free_bytes()
    if free >= GPU_VRAM_REQUIRED_BYTES:
        return "cuda"

    print(
        f"caption: auto → CPU ({free / 1e9:.1f}GB GPU free, "
        f"need ~{GPU_VRAM_REQUIRED_BYTES / 1e9:.1f}GB)",
        file=sys.stderr,
    )
    return "cpu"


def caption_model_loaded() -> bool:
    return _model is not None


def caption_model_device() -> str | None:
    return _model_device


def uses_gpu(device: str) -> bool:
    return device == "cuda"


def unload_caption_model() -> None:
    global _processor, _model, _model_id, _model_device
    _processor = None
    _model = None
    _model_id = None
    _model_device = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _model_dtype(device: str):
    import torch

    return torch.float16 if device == "cuda" else torch.float32


def _move_inputs(inputs, device: str):
    import torch

    dev = torch.device(device)
    dtype = _model_dtype(device)
    moved = {}
    for key, value in inputs.items():
        if value.is_floating_point():
            moved[key] = value.to(dev, dtype=dtype)
        else:
            moved[key] = value.to(dev)
    return moved


def _ensure_model(model_id: str, device: str):
    global _processor, _model, _model_id, _model_device
    if _model is not None and _model_id == model_id and _model_device == device:
        return _processor, _model

    unload_caption_model()
    apply_env()

    from transformers import BlipForConditionalGeneration, BlipProcessor

    dtype = _model_dtype(device)
    print(f"loading caption model {model_id} on {device} ({dtype})...", file=sys.stderr)
    _processor = BlipProcessor.from_pretrained(model_id)
    _model = BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)
    _model_id = model_id
    _model_device = device
    return _processor, _model


def _strip_prompt_prefix(text: str, prompt: str) -> str:
    text = text.strip()
    if prompt and text.lower().startswith(prompt.lower()):
        return text[len(prompt) :].strip()
    return text


def caption_image(
    path: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 120,
    device: CaptionDevicePref | str = "auto",
    pipeline_loaded: bool = False,
) -> str:
    runtime_device = resolve_caption_device(device, pipeline_loaded=pipeline_loaded)
    image = _load_image(path)
    processor, model_obj = _ensure_model(model, runtime_device)

    inputs = _move_inputs(processor(image, prompt, return_tensors="pt"), runtime_device)
    with __import__("torch").inference_mode():
        out = model_obj.generate(
            **inputs,
            max_new_tokens=max_tokens,
            num_beams=5,
            min_length=30,
            no_repeat_ngram_size=3,
        )
    text = processor.decode(out[0], skip_special_tokens=True)
    text = _strip_prompt_prefix(text, prompt)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _caption_via_model(path: str | Path, *, device: CaptionDevicePref | str) -> str:
    from .worker_client import caption_via_worker, ensure_worker

    ensure_worker()
    return caption_via_worker(path, device=device)


def run_caption(
    path: str | Path,
    *,
    device: CaptionDevicePref | str = "auto",
    force_caption: bool = False,
    embed_prompt: bool = False,
) -> str:
    """Extract prompt via metadata or the warm worker."""
    apply_env()
    resolved = Path(path).expanduser().resolve()

    if not force_caption:
        cached = read_prompt_metadata(resolved)
        if cached:
            print("caption: from metadata", file=sys.stderr)
            if embed_prompt:
                write_prompt_metadata(resolved, cached)
            return cached

    prompt = _caption_via_model(resolved, device=device)
    if embed_prompt:
        write_prompt_metadata(resolved, prompt)
    return prompt


def run_captions(
    paths: list[str | Path],
    *,
    device: CaptionDevicePref | str = "auto",
    force_caption: bool = False,
    embed_prompt: bool = False,
) -> None:
    for i, path in enumerate(paths):
        prompt = run_caption(
            path,
            device=device,
            force_caption=force_caption,
            embed_prompt=embed_prompt,
        )
        if i:
            print()
        print(path)
        print()
        print(prompt)
