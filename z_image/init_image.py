from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_init_image(path: Path | str, *, width: int, height: int) -> Image.Image:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"init image not found: {resolved}")
    with Image.open(resolved) as im:
        image = im.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return image.copy()
