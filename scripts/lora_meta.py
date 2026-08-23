#!/usr/bin/env python3
"""Fetch Civitai example resolution (and trigger) into lib/loras.json."""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from z_image.config import LORAS_JSON
from z_image.loras import load_catalog, normalize_filename
from z_image.sizes import snap_dim

_VERSION_RE = re.compile(r"modelVersionId=(\d+)")
_API = "https://civitai.com/api/v1/model-versions/{version_id}"


def parse_version_id(url: str | None) -> int | None:
    if not url:
        return None
    match = _VERSION_RE.search(url)
    return int(match.group(1)) if match else None


def fetch_version_meta(version_id: int) -> dict:
    req = urllib.request.Request(
        _API.format(version_id=version_id),
        headers={"User-Agent": "z-image"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    dims = [
        (int(img["width"]), int(img["height"]))
        for img in data.get("images", [])
        if img.get("width") and img.get("height")
    ]
    width = height = None
    if dims:
        width, height = Counter(dims).most_common(1)[0][0]
        width, height = snap_dim(width), snap_dim(height)
    words = data.get("trainedWords") or []
    trigger = ", ".join(w.strip() for w in words if w and w.strip()) or None
    return {
        "width": width,
        "height": height,
        "trigger": trigger,
        "name": (data.get("model") or {}).get("name"),
        "version": data.get("name"),
    }


def update_entry(entry: dict, meta: dict, *, set_trigger: bool) -> list[str]:
    changes: list[str] = []
    if meta.get("width") and meta.get("height"):
        entry["width"] = meta["width"]
        entry["height"] = meta["height"]
        changes.append(f"size {meta['width']}x{meta['height']}")
    if set_trigger and meta.get("trigger") and not (entry.get("trigger") or "").strip():
        entry["trigger"] = meta["trigger"]
        changes.append(f"trigger {meta['trigger']!r}")
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lora", nargs="*", help="catalog key or stem; omit with --all")
    parser.add_argument("--all", action="store_true", help="update every entry with a Civitai url")
    parser.add_argument("--set-trigger", action="store_true", help="fill empty trigger from Civitai")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    catalog = load_catalog()
    if not catalog:
        raise SystemExit(f"missing or empty {LORAS_JSON}")

    if args.all:
        keys = sorted(catalog.keys())
    elif args.lora:
        keys = []
        for spec in args.lora:
            norm = normalize_filename(spec)
            if norm in catalog:
                keys.append(norm)
            elif norm.removesuffix(".safetensors") in catalog:
                keys.append(norm.removesuffix(".safetensors"))
            else:
                raise SystemExit(f"unknown lora: {spec}")
    else:
        parser.error("pass lora name(s) or --all")

    updated = 0
    for key in keys:
        entry = catalog[key]
        version_id = parse_version_id(entry.get("url"))
        if not version_id:
            print(f"{key}: no civitai url, skipped")
            continue
        try:
            meta = fetch_version_meta(version_id)
        except (OSError, urllib.error.URLError, ValueError, KeyError) as e:
            print(f"{key}: fetch failed ({e})")
            continue
        changes = update_entry(entry, meta, set_trigger=args.set_trigger)
        if not changes:
            print(f"{key}: nothing to update")
            continue
        label = meta.get("version") or meta.get("name") or str(version_id)
        print(f"{key}: {label} → {', '.join(changes)}")
        updated += 1

    if updated and not args.dry_run:
        LORAS_JSON.write_text(json.dumps(catalog, indent=2) + "\n")
        print(f"wrote {LORAS_JSON}")
    elif args.dry_run:
        print("dry run — not writing")


if __name__ == "__main__":
    main()
