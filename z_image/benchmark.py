from __future__ import annotations

from .batch import run_batch
from .config import LORAS_JSON, apply_env
from .log import log
from .loras import benchmark_plan, load_catalog


def run_benchmark(
    *,
    filters: list[str] | None = None,
    seed_base: int = 401,
    repeat: int = 1,
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
        log(f"=== {lora_file} (s{seed}) ===")
        run_batch(
            [prompt],
            loras=[lora_file],
            seeds=[seed],
            seed_set=True,
            repeat=repeat,
            combo=True,
            each=False,
            override=override,
        )
