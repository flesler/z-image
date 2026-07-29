from __future__ import annotations

import re

from .config import SLUG_MAX

_slug_re = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = SLUG_MAX) -> str:
    slug = _slug_re.sub("-", text.lower()).strip("-")
    return slug[:max_len]


def compare_stem(prompt: str, width: int, height: int, seed: int) -> str:
    return f"{slugify(prompt)}--{width}x{height}--s{seed}"


def compare_filename(stem: str, variant: str) -> str:
    return f"{stem}-{variant}.png"
