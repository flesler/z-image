from __future__ import annotations

import re
from typing import NamedTuple

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
VAE_ALIGN = 16

# Reference width for --aspect-ratio (1080 → 1088 for VAE alignment).
ASPECT_BASE_CHOICES = (1024, 1080)


class SizePreset(NamedTuple):
    width: int
    height: int
    label: str


# Widths snapped to multiples of 16 for Z-Image VAE (near Instagram's 1080px targets).
SIZE_PRESETS: dict[str, SizePreset] = {
    "ig-sq": SizePreset(1088, 1088, "Instagram square 1:1"),
    "ig-port": SizePreset(1088, 1360, "Instagram portrait 4:5"),
    "ig-land": SizePreset(1088, 576, "Instagram landscape ~1.91:1"),
    "ig-reel": SizePreset(1088, 1920, "Instagram reel/story 9:16"),
}

_ASPECT_RE = re.compile(r"^(\d+)\s*[:x/]\s*(\d+)$", re.IGNORECASE)


def size_choices() -> list[str]:
    return sorted(SIZE_PRESETS)


def snap_dim(value: float) -> int:
    return max(VAE_ALIGN, round(value / VAE_ALIGN) * VAE_ALIGN)


def require_vae_aligned(value: int, *, name: str = "dimension") -> int:
    value = int(value)
    if value % VAE_ALIGN != 0:
        raise ValueError(f"{name} must be divisible by {VAE_ALIGN} (got {value})")
    return value


def resolve_aspect_base(base: int) -> int:
    if base == 1080:
        return snap_dim(1080)
    if base == 1024:
        return 1024
    valid = ", ".join(str(v) for v in ASPECT_BASE_CHOICES)
    raise SystemExit(f"--aspect-base must be one of: {valid}")


def parse_aspect_ratio(value: str) -> tuple[int, int]:
    match = _ASPECT_RE.match(value.strip())
    if not match:
        raise SystemExit(f"invalid --aspect-ratio {value!r}; use W:H e.g. 4:5, 16:9, 9:16")
    ratio_w = int(match.group(1))
    ratio_h = int(match.group(2))
    if ratio_w < 1 or ratio_h < 1:
        raise SystemExit("--aspect-ratio values must be positive integers")
    return ratio_w, ratio_h


def dimensions_from_aspect(ratio: str, *, aspect_base: int) -> tuple[int, int]:
    ratio_w, ratio_h = parse_aspect_ratio(ratio)
    width = resolve_aspect_base(aspect_base)
    height = snap_dim(width * ratio_h / ratio_w)
    return width, height


def resolve_dimensions(
    *,
    size: str | None,
    aspect_ratio: str | None,
    aspect_base: int,
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    if aspect_base not in ASPECT_BASE_CHOICES:
        valid = ", ".join(str(v) for v in ASPECT_BASE_CHOICES)
        raise SystemExit(f"--aspect-base must be one of: {valid}")
    if aspect_ratio is None and aspect_base != 1024:
        raise SystemExit("--aspect-base requires --aspect-ratio")

    dim_flags = [
        size is not None,
        aspect_ratio is not None,
        width is not None or height is not None,
    ]
    if sum(dim_flags) > 1:
        raise SystemExit("--size, --aspect-ratio, and --width/--height are mutually exclusive")

    if size is not None:
        preset = SIZE_PRESETS.get(size)
        if preset is None:
            valid = ", ".join(size_choices())
            raise SystemExit(f"unknown --size {size!r}; choose from: {valid}")
        return preset.width, preset.height

    if aspect_ratio is not None:
        return dimensions_from_aspect(aspect_ratio, aspect_base=aspect_base)

    if width is None and height is None:
        return DEFAULT_WIDTH, DEFAULT_HEIGHT
    if width is None or height is None:
        raise SystemExit("provide both --width and --height, or use --size / --aspect-ratio")
    return snap_dim(width), snap_dim(height)
