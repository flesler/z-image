from __future__ import annotations

import sys
from pathlib import Path

from .config import apply_env, output_dir
from .loras import LoraSpec, apply_triggers, normalize_prompt, random_seed, resolve_spec
from .naming import output_filename
from .worker_client import ensure_worker, generate_cold, generate_via_worker


def run_generate(
    prompt: str,
    *,
    loras: list[str] | None = None,
    precision: str = "q4",
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    steps: int = 9,
    output: Path | None = None,
    cold: bool = False,
    extra: list[str] | None = None,
) -> Path:
    apply_env()
    prompt = normalize_prompt(prompt)
    loras = loras or []
    seed = seed if seed is not None else random_seed()

    if output is None:
        out_dir = output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / output_filename(prompt, width, height, seed)

    print(f"seed: {seed}", file=sys.stderr)
    print(f"→ {output}", file=sys.stderr)

    resolved: list[LoraSpec] = [resolve_spec(spec) for spec in loras]
    final_prompt = apply_triggers(prompt, loras) if loras else prompt
    if loras:
        print(f"prompt: {final_prompt}", file=sys.stderr)

    if cold:
        generate_cold(
            prompt=final_prompt,
            output=output,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            precision=precision,
            loras=resolved,
            extra=extra,
        )
    else:
        ensure_worker(cold=False)
        generate_via_worker(
            prompt=final_prompt,
            output=output,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            precision=precision,
            loras=resolved,
        )

    return output
