from __future__ import annotations

import re

from .config import resolve_steps, SLUG_MAX

_slug_re = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = SLUG_MAX) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    slug = _slug_re.sub("-", text.lower()).strip("-")
    return slug[:max_len]


def lora_slug(names: list[str] | None) -> str | None:
    if not names:
        return None
    from .loras import lora_name

    stems = [lora_name(name) for name in names]
    if len(stems) == 1:
        return slugify(stems[0])
    return slugify("+".join(stems))


def batch_stem(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int | None = None,
    *,
    strength: float | None = None,
    lora_names: list[str] | None = None,
) -> str:
    resolved = resolve_steps(steps)
    parts = [slugify(prompt), f"{width}x{height}"]
    lora = lora_slug(lora_names)
    if lora:
        parts.append(lora)
    parts.append(str(seed))
    base = "-".join(parts)
    if resolved != resolve_steps(None):
        base += f"-s{resolved}"
    if strength is not None:
        base += f"-i{strength:g}"
    return base


def output_filename(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int | None = None,
    *,
    strength: float | None = None,
    lora_names: list[str] | None = None,
) -> str:
    return f"{batch_stem(prompt, width, height, seed, steps, strength=strength, lora_names=lora_names)}.png"


def batch_list_glob(prompt: str, width: int, height: int) -> str:
    """Glob pattern to list all variants for a prompt/size (seed wildcard)."""
    return f"{slugify(prompt)}-{width}x{height}-*"


def batch_filename(stem: str, variant: str) -> str:
    return f"{stem}-{variant}.png"
