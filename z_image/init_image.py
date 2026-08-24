from __future__ import annotations

from pathlib import Path

from PIL import Image


def fit_output_size(image: Image.Image, width: int, height: int) -> Image.Image:
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def load_init_image(path: Path | str, *, width: int, height: int) -> Image.Image:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"init image not found: {resolved}")
    with Image.open(resolved) as im:
        image = im.convert("RGB")
        return fit_output_size(image, width, height).copy()
