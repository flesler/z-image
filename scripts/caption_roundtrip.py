#!/usr/bin/env python3
"""Regenerate from extracted captions to test round-trip quality."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "outputs" / "caption-eval"
RESULTS = EVAL_DIR / "caption-results.json"
CLI = ROOT / "cli.py"


def main() -> int:
    if not RESULTS.is_file():
        print(f"run caption_eval.py first; missing {RESULTS}", file=sys.stderr)
        return 1

    results = json.loads(RESULTS.read_text())
    for i, row in enumerate(results, 1):
        w, h = row["size"].split("x")
        out = EVAL_DIR / f"roundtrip-{i}-from-caption.png"
        prompt = row["extracted"]
        print(f"\n=== roundtrip {i}: {out.name} ===", file=sys.stderr)
        print(f"prompt: {prompt}", file=sys.stderr)
        subprocess.run(
            [
                str(CLI),
                "gen",
                prompt,
                "--seed",
                "2001",
                "-w",
                w,
                "-H",
                h,
                "-o",
                str(out),
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
