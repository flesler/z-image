from __future__ import annotations

import sys

from .compare import run_compare
from .config import LORAS_JSON, apply_env
from .loras import benchmark_plan, load_catalog


def run_benchmark(
    *,
    filters: list[str] | None = None,
    seed_base: int = 401,
    repeat: int = 1,
    cold: bool = False,
    override: bool = False,
) -> None:
    apply_env()
    if not LORAS_JSON.is_file():
        raise SystemExit(f"missing {LORAS_JSON}")

    catalog = load_catalog()
    plan = benchmark_plan(catalog, seed_base, repeat, filters)
    if not plan:
        raise SystemExit("no LoRAs with prompts to benchmark")

    for lora_file, prompt, seed in plan:
        print(f"=== {lora_file} (s{seed}) ===", file=sys.stderr)
        run_compare(
            prompt,
            loras=[lora_file],
            seed=seed,
            seed_set=True,
            repeat=repeat,
            combo=True,
            each=False,
            cold=cold,
            override=override,
        )
