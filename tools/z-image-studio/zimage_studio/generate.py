from __future__ import annotations

import sys
from pathlib import Path

from .config import apply_env, output_dir, resolve_strength, validate_strength
from .init_image import load_init_image
from .loras import LoraSpec, apply_triggers, normalize_prompt, random_seed, resolve_spec
from .naming import output_filename
from .worker_client import ensure_worker, generate_via_worker


def run_generate(
    prompt: str,
    *,
    loras: list[str] | None = None,
    precision: str | None = None,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    steps: int | None = None,
    output: Path | None = None,
    image: Path | None = None,
    strength: float | None = None,
    extra: list[str] | None = None,
) -> Path:
    apply_env()
    prompt = normalize_prompt(prompt)
    loras = loras or []
    seed = seed if seed is not None else random_seed()
    img_strength = None
    if image is not None:
        if strength is not None:
            validate_strength(strength)
        img_strength = resolve_strength(strength)
        load_init_image(image, width=width, height=height)

    if output is None:
        out_dir = output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / output_filename(
            prompt,
            width,
            height,
            seed,
            steps,
            strength=img_strength,
        )

    print(f"seed: {seed}", file=sys.stderr)
    if image is not None:
        print(f"img2img: {image} strength={img_strength:g}", file=sys.stderr)
    print(f"→ {output}", file=sys.stderr)

    resolved: list[LoraSpec] = [resolve_spec(spec) for spec in loras]
    final_prompt = apply_triggers(prompt, loras) if loras else prompt
    if loras:
        print(f"prompt: {final_prompt}", file=sys.stderr)

    ensure_worker()
    generate_via_worker(
        prompt=final_prompt,
        output=output,
        width=width,
        height=height,
        seed=seed,
        steps=steps,
        precision=precision,
        loras=resolved,
        image=image,
        strength=img_strength,
    )

    return output
