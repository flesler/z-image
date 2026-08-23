from __future__ import annotations

from pathlib import Path

from .config import apply_env, output_dir, resolve_strength, validate_strength
from .init_image import load_init_image
from .log import log
from .loras import LoraSpec, normalize_prompt, random_seed, resolve_spec
from .naming import output_filename
from .templates import resolve_prompt
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
    resolved = resolve_prompt(prompt, seed)
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
            resolved.prompt,
            width,
            height,
            seed,
            steps,
            strength=img_strength,
        )

    log(f"seed: {seed}")
    if image is not None:
        log(f"img2img: {image} strength={img_strength:g}")
    log(f"→ {output}")

    resolved_loras: list[LoraSpec] = [resolve_spec(spec) for spec in loras]
    if loras:
        log(f"prompt: {resolved.prompt}")

    ensure_worker()
    generate_via_worker(
        prompt=prompt,
        output=output,
        width=width,
        height=height,
        seed=seed,
        seed_base=seed,
        steps=steps,
        precision=precision,
        loras=resolved_loras,
        image=image,
        strength=img_strength,
    )

    return output
