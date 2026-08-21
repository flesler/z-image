from __future__ import annotations

import sys
from pathlib import Path

from .batch import dedupe_ints, run_batch
from .caption import image_dimensions, run_caption
from .config import apply_env
from .generate import run_generate
from .loras import random_seed


def run_from_image(
    image: Path,
    *,
    loras: list[str] | None = None,
    precision: str | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    seeds: list[int] | None = None,
    repeat: int = 1,
    steps: int | None = None,
    output: Path | None = None,
    caption_only: bool = False,
    caption_device: str = "auto",
    force_caption: bool = False,
    embed_prompt: bool = False,
) -> str:
    apply_env()
    img_w, img_h = image_dimensions(image)
    gen_w = width if width is not None else img_w
    gen_h = height if height is not None else img_h

    prompt = run_caption(
        image,
        device=caption_device,
        force_caption=force_caption,
        embed_prompt=embed_prompt,
    )
    print(f"caption: {prompt}", file=sys.stderr)
    print(f"size: {gen_w}x{gen_h} (source {img_w}x{img_h})", file=sys.stderr)

    if caption_only:
        print(prompt)
        return prompt

    if repeat > 1 or (seeds and len(seeds) > 1):
        resolved_seeds = dedupe_ints(seeds)
        run_batch(
            [prompt],
            loras=loras or [],
            seeds=resolved_seeds,
            seed_set=bool(resolved_seeds),
            repeat=repeat,
            width=gen_w,
            height=gen_h,
            steps_list=[steps] if steps is not None else None,
            precision=precision,
            each=bool(loras),
        )
        return prompt

    run_generate(
        prompt,
        loras=loras,
        precision=precision,
        width=gen_w,
        height=gen_h,
        seed=seed if seed is not None else random_seed(),
        steps=steps,
        output=output,
    )
    return prompt
