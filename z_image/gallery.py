from __future__ import annotations

import os
from pathlib import Path

from .config import data_output_dir

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"})


def gallery_root() -> Path:
    return data_output_dir().expanduser().resolve()


def resolve_gallery_dir(subfolder: str | None = None) -> Path:
    if not subfolder:
        return gallery_root()
    return resolve_gallery_path(subfolder)


def resolve_gallery_path(relative: str) -> Path:
    root = gallery_root()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("invalid path")

    resolved = (root / rel).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("invalid path") from None
    return resolved


def open_gallery_file(relative: str) -> Path:
    path = resolve_gallery_path(relative)
    if not path.is_file():
        raise FileNotFoundError(relative)
    return path


def _created_at(stat: os.stat_result) -> float:
    return float(getattr(stat, "st_birthtime", stat.st_mtime))


def _append_image(root: Path, entry: os.DirEntry[str], images: list[dict]) -> None:
    if not entry.is_file(follow_symlinks=False):
        return
    suffix = Path(entry.name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        return
    stat = entry.stat(follow_symlinks=False)
    rel = Path(entry.path).resolve().relative_to(root)
    images.append(
        {
            "name": entry.name,
            "path": rel.as_posix(),
            "created_at": _created_at(stat),
            "size": stat.st_size,
        }
    )


def _scan_images(directory: Path, root: Path, images: list[dict]) -> None:
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                if entry.name.startswith("."):
                    continue
                _scan_images(Path(entry.path), root, images)
            else:
                _append_image(root, entry, images)


def list_images(subfolder: str | None = None, *, recursive: bool = False) -> list[dict]:
    """List image files in the gallery root or a subfolder, newest first."""
    root = gallery_root()
    directory = resolve_gallery_dir(subfolder)
    if not directory.is_dir():
        return []

    images: list[dict] = []
    if recursive:
        _scan_images(directory, root, images)
    else:
        with os.scandir(directory) as entries:
            for entry in entries:
                _append_image(root, entry, images)

    images.sort(key=lambda item: item["created_at"], reverse=True)
    return images
