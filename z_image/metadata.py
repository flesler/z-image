"""Stable Diffusion-compatible PNG metadata (parameters text chunk)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .loras import lora_name

_FILENAME_SIZE_RE = re.compile(r"-(\d+)x(\d+)-", re.I)
_FILENAME_SEED_RE = re.compile(r"-(\d+)(?:-s\d+)?(?:-i[\d.]+)?(?:-base)?\.[a-z]+$", re.I)

MODEL_NAME = "Z-Image-Turbo"
SAMPLER = "Flow Match"
NEGATIVE_PROMPT_MARKER = "Negative prompt:"
TEMPLATE_MARKER = "Template:"


@dataclass(frozen=True)
class GenMeta:
    prompt: str
    width: int
    height: int
    seed: int
    steps: int
    template: str | None = None
    precision: str | None = None
    loras: list[tuple[str, float]] | None = None
    strength: float | None = None


def _lora_names(loras: list[tuple[str, float]]) -> str:
    names: list[str] = []
    for path_or_name, _ in loras:
        name = lora_name(path_or_name)
        if name and name not in names:
            names.append(name)
    return ", ".join(names)


def build_parameters(meta: GenMeta) -> str:
    lines = [meta.prompt]
    if meta.template:
        lines.append(f"{TEMPLATE_MARKER} {meta.template}")
    lines.append("Negative prompt: ")
    parts = [
        f"Steps: {meta.steps}",
        f"Sampler: {SAMPLER}",
        "CFG scale: 0",
        f"Seed: {meta.seed}",
        f"Size: {meta.width}x{meta.height}",
        f"Model: {MODEL_NAME}",
    ]
    if meta.precision:
        parts.append(f"Precision: {meta.precision}")
    if meta.loras:
        parts.append(f"LoRA: {_lora_names(meta.loras)}")
    if meta.strength is not None:
        parts.append(f"Denoising strength: {meta.strength:g}")
    lines.append(", ".join(parts))
    return "\n".join(lines)


def save_image(image: Image.Image, path: Path | str, meta: GenMeta) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    pnginfo = PngInfo()
    pnginfo.add_text("parameters", build_parameters(meta))
    image.save(resolved, pnginfo=pnginfo)


def _image_text(im: Image.Image) -> dict[str, str]:
    text = getattr(im, "text", None)
    return text if isinstance(text, dict) else {}


def meta_from_filename(name: str) -> dict | None:
    size_match = _FILENAME_SIZE_RE.search(name)
    if not size_match:
        return None
    seed_match = _FILENAME_SEED_RE.search(name)
    if not seed_match:
        return None
    return {
        "width": int(size_match.group(1)),
        "height": int(size_match.group(2)),
        "seed": int(seed_match.group(1)),
    }


def parse_parameters(raw: str) -> dict | None:
    prompt_end = raw.find(NEGATIVE_PROMPT_MARKER)
    if prompt_end < 0:
        return None
    header = raw[:prompt_end].rstrip()
    header_lines = header.split("\n")
    prompt = header_lines[0].strip()
    template = None
    for line in header_lines[1:]:
        if line.startswith(TEMPLATE_MARKER):
            template = line[len(TEMPLATE_MARKER) :].strip()
            break
    info = raw[prompt_end:]

    def field(name: str) -> str | None:
        match = re.search(rf"{name}:\s*([^,\n]+)", info, re.I)
        return match.group(1).strip() if match else None

    size = field("Size")
    width = height = None
    if size and "x" in size:
        parts = size.split("x", 1)
        try:
            width = int(parts[0])
            height = int(parts[1])
        except ValueError:
            width = height = None
    seed_val = field("Seed")
    steps_val = field("Steps")
    lora_raw = field("LoRA")
    lora_known = re.search(r"(?:^|[\n,])\s*LoRA:\s*", info, re.I) is not None
    return {
        "prompt": prompt or None,
        "template": template,
        "width": width,
        "height": height,
        "seed": int(seed_val) if seed_val is not None else None,
        "steps": int(steps_val) if steps_val is not None else None,
        "lora": lora_raw.split(",", 1)[0].strip() if lora_raw else None,
        "loraKnown": lora_known,
    }


def read_gen_meta(path: Path | str) -> dict | None:
    """Generation metadata from filename + PNG parameters (CPU-only, no GPU)."""
    resolved = Path(path).expanduser().resolve()
    from_name = meta_from_filename(resolved.name) or {}
    try:
        with Image.open(resolved) as im:
            raw = _image_text(im).get("parameters")
    except OSError:
        raw = None
    if not raw:
        return from_name or None
    from_png = parse_parameters(raw)
    if not from_png:
        return from_name or None
    merged = {
        **from_name,
        **from_png,
        "prompt": from_png.get("prompt") or from_name.get("prompt"),
        "seed": from_png.get("seed") if from_png.get("seed") is not None else from_name.get("seed"),
        "steps": from_png.get("steps"),
    }
    if from_png.get("loraKnown"):
        merged["lora"] = from_png.get("lora")
        merged["loraKnown"] = True
    return merged


def read_prompt(path: Path | str) -> str | None:
    """Positive prompt from PNG parameters chunk, or None."""
    resolved = Path(path).expanduser().resolve()
    with Image.open(resolved) as im:
        raw = _image_text(im).get("parameters")
    if not raw:
        return None
    if NEGATIVE_PROMPT_MARKER in raw:
        prompt = raw.split(NEGATIVE_PROMPT_MARKER, 1)[0]
    else:
        prompt = raw.split("\n", 1)[0]
    prompt = prompt.strip()
    return prompt or None


def embed_prompt(path: Path | str, prompt: str) -> None:
    """Write prompt into PNG parameters (mutates file)."""
    resolved = Path(path).expanduser().resolve()
    with Image.open(resolved) as im:
        image = im.convert("RGB")
        extra = {k: v for k, v in _image_text(im).items() if k != "parameters"}
    width, height = image.size
    pnginfo = PngInfo()
    for key, value in extra.items():
        pnginfo.add_text(key, value)
    pnginfo.add_text(
        "parameters",
        build_parameters(GenMeta(prompt=prompt, width=width, height=height, seed=0, steps=0)),
    )
    image.save(resolved, pnginfo=pnginfo)
