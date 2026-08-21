#!/usr/bin/env python3
"""Evaluate image-to-prompt on z-image outputs with known ground-truth prompts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from z_image.caption import caption_image, image_dimensions, unload_caption_model

EVAL_DIR = ROOT / "data" / "outputs" / "caption-eval"
PROMPTS_FILE = EVAL_DIR / "prompts.json"


def main() -> int:
    ground_truth = json.loads(PROMPTS_FILE.read_text())
    results: list[dict] = []

    for name, original in ground_truth.items():
        path = EVAL_DIR / name
        w, h = image_dimensions(path)
        extracted = caption_image(path)
        results.append(
            {
                "file": name,
                "size": f"{w}x{h}",
                "original": original,
                "extracted": extracted,
            }
        )
        print(f"\n{'=' * 72}\n{name} ({w}x{h})\n{'=' * 72}")
        print(f"ORIGINAL:\n  {original}\n")
        print(f"EXTRACTED:\n  {extracted}\n")

    out = EVAL_DIR / "caption-results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n→ {out}", file=sys.stderr)
    unload_caption_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
