from __future__ import annotations

import sys
from pathlib import Path

from .config import PREVIEW_STEPS, apply_env, compare_dir, resolve_steps
from .loras import LoraSpec, apply_triggers, expand_lora_specs, normalize_prompt, random_seed, resolve_spec
from .naming import compare_filename, compare_stem
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


def normalize_steps_list(values: list[int] | None) -> list[int] | None:
    steps = dedupe_ints(values)
    if not steps:
        return None
    for step in steps:
        if step < 1:
            raise SystemExit("--steps values must be >= 1")
    return steps


def expand_seed_runs(seeds: list[int] | None, repeat: int, *, seed_set: bool) -> list[int]:
    if repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    if seed_set:
        if not seeds:
            raise SystemExit("need at least one --seed value")
        return [s + i for s in seeds for i in range(repeat)]
    base = random_seed()
    return [base + i for i in range(repeat)]


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
    precision: str | None = None,
    each: bool = False,
    combo: bool = False,
    include_base: bool = True,
    reuse_steps: bool = False,
    preview: bool = False,
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

    default_steps = resolve_steps(None)
    if preview:
        steps_list = [PREVIEW_STEPS]
        print(f"preview: {PREVIEW_STEPS} steps", file=sys.stderr)
    else:
        steps_list = normalize_steps_list(steps_list) or [default_steps]
        if len(steps_list) > 1 or steps_list[0] != default_steps:
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

    seed_runs = expand_seed_runs(seeds, repeat, seed_set=seed_set)
    if len(seed_runs) == 1:
        suffix = " (random)" if not seed_set else ""
        print(f"seed: {seed_runs[0]}{suffix}", file=sys.stderr)
    else:
        if not seed_set:
            print(f"seed: {seed_runs[0]} (random)", file=sys.stderr)
        print(f"seeds: {', '.join(str(s) for s in seed_runs)}", file=sys.stderr)

    outputs: list[Path] = []
    outputs.extend(
        _run_one_compare(
            prompts=prompts,
            resolved=resolved,
            seed_runs=seed_runs,
            root=root,
            width=width,
            height=height,
            steps_list=steps_list,
            precision=precision,
            each=each,
            combo=combo,
            include_base=include_base,
            reuse_steps=reuse_steps,
            preview=preview,
            override=override,
            dry_run=dry_run,
        )
    )

    return outputs


def _log_generated(path: str | Path, elapsed_s: float, *, partial: bool = False) -> None:
    tag = " partial" if partial else ""
    print(f"{path} {elapsed_s:.1f}s{tag}", file=sys.stderr, flush=True)


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
    seed_runs: list[int],
    root: Path,
    width: int,
    height: int,
    steps_list: list[int],
    precision: str,
    each: bool,
    combo: bool,
    include_base: bool,
    reuse_steps: bool,
    preview: bool,
    override: bool,
    dry_run: bool,
) -> list[Path]:
    outputs: list[Path] = []
    jobs: list[dict] = []
    models = _model_variants(resolved, each=each, combo=combo, include_base=include_base)
    if not models:
        raise SystemExit("no model variants to run (use --lora and/or drop --no-base)")

    for iter_seed in seed_runs:
        if len(seed_runs) > 1:
            print(f"seed run: {iter_seed}", file=sys.stderr)

        for model_label, lora_specs, variant in models:
            for prompt in prompts:
                for step_count in steps_list:
                    final_prompt = apply_triggers(prompt, [str(spec) for spec in lora_specs])
                    name_steps = None if preview else step_count
                    stem = compare_stem(prompt, width, height, iter_seed, name_steps)
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
        precision=precision,
        reuse_steps=reuse_steps,
        on_image=lambda r: _log_generated(
            r["output"],
            r["elapsed_s"],
            partial=reuse_steps,
        ),
    )

    return outputs
