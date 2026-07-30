from __future__ import annotations

import sys
from pathlib import Path

from .config import apply_env, compare_dir
from .loras import LoraSpec, apply_triggers, expand_lora_specs, normalize_prompt, random_seed, resolve_spec
from .naming import DEFAULT_COMPARE_STEPS, compare_filename, compare_stem
from .worker_client import ensure_worker, generate_batch_via_worker


def collect_prompts(*, inline: list[str] | None, files: list[Path]) -> list[str]:
    """Merge --prompt and --prompt-file lines; skip empty and duplicates (order preserved)."""
    prompts: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        text = normalize_prompt(raw)
        if not text or text in seen:
            return
        seen.add(text)
        prompts.append(text)

    for prompt in inline or []:
        add(prompt)
    for path in files:
        if not path.is_file():
            raise SystemExit(f"prompt file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            add(line)
    if not prompts:
        raise SystemExit("need at least one --prompt or --prompt-file")
    return prompts


def dedupe_ints(values: list[int] | None) -> list[int]:
    """Dedupe int list, preserve order."""
    if not values:
        return []
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def normalize_steps_list(values: list[int] | None) -> list[int]:
    steps = dedupe_ints(values)
    if not steps:
        return [DEFAULT_COMPARE_STEPS]
    for step in steps:
        if step < 1:
            raise SystemExit("--steps values must be >= 1")
    return steps


def run_compare(
    prompts: list[str],
    *,
    loras: list[str],
    seeds: list[int] | None = None,
    seed_set: bool = False,
    repeat: int = 1,
    width: int = 1024,
    height: int = 1024,
    steps_list: list[int] | None = None,
    precision: str = "q4",
    each: bool = False,
    combo: bool = False,
    include_base: bool = True,
    each_step: bool = False,
    override: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    apply_env()
    prompts = [normalize_prompt(p) for p in prompts]
    if not prompts:
        raise SystemExit("need at least one --prompt")
    if not loras and not include_base:
        raise SystemExit("need --lora or drop --no-base")
    if repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    steps_list = steps_list or [DEFAULT_COMPARE_STEPS]
    if len(steps_list) > 1 or steps_list[0] != DEFAULT_COMPARE_STEPS:
        print(f"steps: {', '.join(str(s) for s in steps_list)}", file=sys.stderr)

    wildcard = loras and any(s.split(":", 1)[0].strip().lower() in ("*", "all") for s in loras)
    loras = expand_lora_specs(loras) if loras else []
    if wildcard:
        print(f"loras: expanded '*' → {len(loras)} adapter(s)", file=sys.stderr)

    resolved = [resolve_spec(spec) for spec in loras]
    if not each and not combo:
        each = True
        combo = len(resolved) > 1

    root = compare_dir()
    root.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        ensure_worker()

    seed_runs = seeds if seed_set else [None]
    if seed_set and not seeds:
        raise SystemExit("need at least one --seed value")
    if seed_set and len(seeds) > 1:
        print(f"seeds: {', '.join(str(s) for s in seeds)}", file=sys.stderr)

    outputs: list[Path] = []

    for rep in range(repeat):
        if repeat > 1:
            print(f"repeat {rep + 1}/{repeat}", file=sys.stderr)
        outputs.extend(
            _run_one_compare(
                prompts=prompts,
                resolved=resolved,
                seed_runs=seed_runs,
                seed_set=seed_set,
                root=root,
                width=width,
                height=height,
                steps_list=steps_list,
                precision=precision,
                each=each,
                combo=combo,
                include_base=include_base,
                each_step=each_step,
                override=override,
                dry_run=dry_run,
            )
        )

    return outputs


def _log_generated(path: str | Path, elapsed_s: float) -> None:
    print(f"{path} {elapsed_s:.1f}s", file=sys.stderr, flush=True)


def _model_variants(
    resolved: list[LoraSpec],
    *,
    each: bool,
    combo: bool,
    include_base: bool = True,
) -> list[tuple[str, list[LoraSpec], str]]:
    models: list[tuple[str, list[LoraSpec], str]] = []
    if include_base:
        models.append(("base", [], "base"))
    if each:
        for spec in resolved:
            models.append((f"lora {spec}", [spec], spec.name))
    if combo and len(resolved) > 1:
        combo_name = "_combo" + "".join(f"+{spec.name}" for spec in resolved)
        models.append((f"lora combo ({len(resolved)})", resolved, combo_name))
    elif combo and len(resolved) == 1 and not each:
        spec = resolved[0]
        models.append((f"lora {spec}", [spec], spec.name))
    return models


def _run_one_compare(
    *,
    prompts: list[str],
    resolved: list[LoraSpec],
    seed_runs: list[int | None],
    seed_set: bool,
    root: Path,
    width: int,
    height: int,
    steps_list: list[int],
    precision: str,
    each: bool,
    combo: bool,
    include_base: bool,
    each_step: bool,
    override: bool,
    dry_run: bool,
) -> list[Path]:
    outputs: list[Path] = []
    jobs: list[dict] = []
    models = _model_variants(resolved, each=each, combo=combo, include_base=include_base)
    if not models:
        raise SystemExit("no model variants to run (use --lora and/or drop --no-base)")

    for seed_idx, run_seed in enumerate(seed_runs):
        iter_seed = run_seed if seed_set else random_seed()
        if seed_set and len(seed_runs) > 1:
            print(f"seed run: {iter_seed}", file=sys.stderr)
        elif seed_idx == 0:
            suffix = " (random)" if not seed_set else ""
            print(f"seed: {iter_seed}{suffix}", file=sys.stderr)

        for model_label, lora_specs, variant in models:
            for prompt in prompts:
                for step_count in steps_list:
                    final_prompt = apply_triggers(prompt, [str(spec) for spec in lora_specs])
                    stem = compare_stem(prompt, width, height, iter_seed, step_count)
                    output = root / compare_filename(stem, variant)

                    step_label = f" s{step_count}" if len(steps_list) > 1 else ""
                    if not override and output.is_file():
                        print(f"⊘ skip {model_label}{step_label} → {output}", file=sys.stderr)
                        outputs.append(output)
                        continue

                    if dry_run:
                        print(f"→ {model_label}{step_label} → {output}", file=sys.stderr)
                        if final_prompt != prompt:
                            print(f"prompt: {final_prompt}", file=sys.stderr)
                    jobs.append(
                        {
                            "prompt": final_prompt,
                            "output": str(output),
                            "seed": iter_seed,
                            "steps": step_count,
                            "loras": [{"file": spec.file, "strength": spec.strength} for spec in lora_specs],
                        }
                    )
                    outputs.append(output)

    if not jobs or dry_run:
        return outputs

    generate_batch_via_worker(
        jobs=jobs,
        width=width,
        height=height,
        steps=DEFAULT_COMPARE_STEPS,
        precision=precision,
        each_step=each_step,
        on_image=lambda r: _log_generated(r["output"], r["elapsed_s"]),
    )

    return outputs
