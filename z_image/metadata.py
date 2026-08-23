"""Stable Diffusion-compatible PNG metadata (parameters text chunk)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .loras import lora_name

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
