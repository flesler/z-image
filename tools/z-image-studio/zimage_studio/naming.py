from __future__ import annotations

import re

from .config import resolve_steps, SLUG_MAX

_slug_re = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = SLUG_MAX) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    slug = _slug_re.sub("-", text.lower()).strip("-")
    return slug[:max_len]


def batch_stem(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int | None = None,
) -> str:
    resolved = resolve_steps(steps)
    base = f"{slugify(prompt)}-{width}x{height}-{seed}"
    if resolved != resolve_steps(None):
        base += f"-s{resolved}"
    return base


def output_filename(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int | None = None,
) -> str:
    return f"{batch_stem(prompt, width, height, seed, steps)}.png"


def batch_list_glob(prompt: str, width: int, height: int) -> str:
    """Glob pattern to list all variants for a prompt/size (seed wildcard)."""
    return f"{slugify(prompt)}-{width}x{height}-*"


def batch_filename(stem: str, variant: str) -> str:
    return f"{stem}-{variant}.png"
