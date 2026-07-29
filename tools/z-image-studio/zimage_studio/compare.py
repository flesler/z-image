from __future__ import annotations

import sys
from pathlib import Path

from .config import apply_env, compare_dir, load_pipeline
from .generate import run_generate
from .loras import LoraSpec, apply_triggers, expand_lora_specs, normalize_prompt, random_seed, resolve_spec
from .naming import DEFAULT_COMPARE_STEPS, compare_filename, compare_stem
from .pipeline_jobs import run_batch_on_pipe
from .text_encoder import release_text_encoder
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


def parse_steps_list(raw: str) -> list[int]:
    """Parse --steps '7,8,9,10' — skip empty, dedupe, preserve order."""
    values: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        steps = int(part)
        if steps < 1:
            raise SystemExit("--steps values must be >= 1")
        if steps in seen:
            continue
        seen.add(steps)
        values.append(steps)
    if not values:
        raise SystemExit("need at least one --steps value")
    return values


def run_compare(
    prompts: list[str],
    *,
    loras: list[str],
    seed: int | None = None,
    seed_set: bool = False,
    repeat: int = 1,
    width: int = 1024,
    height: int = 1024,
    steps_list: list[int] | None = None,
    precision: str = "q4",
    each: bool = False,
    combo: bool = False,
    cold: bool = False,
    override: bool = False,
) -> list[Path]:
    apply_env()
    prompts = [normalize_prompt(p) for p in prompts]
    if not prompts:
        raise SystemExit("need at least one --prompt")
    if not loras:
        raise SystemExit("need at least one --lora name[:strength]")
    if repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    steps_list = steps_list or [DEFAULT_COMPARE_STEPS]
    if len(steps_list) > 1 or steps_list[0] != DEFAULT_COMPARE_STEPS:
        print(f"steps: {', '.join(str(s) for s in steps_list)}", file=sys.stderr)

    wildcard = any(s.split(":", 1)[0].strip().lower() in ("*", "all") for s in loras)
    loras = expand_lora_specs(loras)
    if wildcard:
        print(f"loras: expanded '*' → {len(loras)} adapter(s)", file=sys.stderr)

    resolved = [resolve_spec(spec) for spec in loras]
    if not each and not combo:
        each = True
        combo = len(resolved) > 1

    root = compare_dir()
    root.mkdir(parents=True, exist_ok=True)

    if not cold:
        ensure_worker(cold=False)

    outputs: list[Path] = []

    for rep in range(repeat):
        if repeat > 1:
            print(f"repeat {rep + 1}/{repeat}", file=sys.stderr)
        outputs.extend(
            _run_one_compare(
                prompts=prompts,
                resolved=resolved,
                rep=rep,
                seed=seed,
                seed_set=seed_set,
                root=root,
                width=width,
                height=height,
                steps_list=steps_list,
                precision=precision,
                each=each,
                combo=combo,
                cold=cold,
                override=override,
            )
        )

    print(f"compare root: {root}", file=sys.stderr)
    for path in outputs:
        print(path)
    return outputs


def _prompt_seed(*, seed: int | None, seed_set: bool, rep: int, prompt_idx: int, n_prompts: int) -> int:
    if seed_set:
        return (seed or 0) + rep * n_prompts + prompt_idx
    return random_seed()


def _model_variants(
    resolved: list[LoraSpec],
    *,
    each: bool,
    combo: bool,
) -> list[tuple[str, list[LoraSpec], str]]:
    models: list[tuple[str, list[LoraSpec], str]] = [("base", [], "base")]
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
    rep: int,
    seed: int | None,
    seed_set: bool,
    root: Path,
    width: int,
    height: int,
    steps_list: list[int],
    precision: str,
    each: bool,
    combo: bool,
    cold: bool,
    override: bool,
) -> list[Path]:
    outputs: list[Path] = []
    jobs: list[dict] = []
    models = _model_variants(resolved, each=each, combo=combo)

    for model_label, lora_specs, variant in models:
        for step_count in steps_list:
            for prompt_idx, prompt in enumerate(prompts):
                iter_seed = _prompt_seed(
                    seed=seed,
                    seed_set=seed_set,
                    rep=rep,
                    prompt_idx=prompt_idx,
                    n_prompts=len(prompts),
                )
                if prompt_idx == 0 and model_label == "base" and step_count == steps_list[0]:
                    print(f"seed: {iter_seed}" + (f" (+1 per prompt)" if len(prompts) > 1 and seed_set else ""), file=sys.stderr)
                final_prompt = apply_triggers(prompt, [str(spec) for spec in lora_specs])
                stem = compare_stem(prompt, width, height, iter_seed, step_count)
                output = root / compare_filename(stem, variant)

                step_label = f" s{step_count}" if len(steps_list) > 1 else ""
                if not override and output.is_file():
                    print(f"⊘ skip {model_label}{step_label} (exists) → {output}", file=sys.stderr)
                    outputs.append(output)
                    continue

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

    if not jobs:
        return outputs

    if cold:
        pipe = load_pipeline(precision=precision)
        try:
            from .loras import resolve_path

            resolved_jobs = []
            for job in jobs:
                loras = [
                    (str(resolve_path(entry["file"])), float(entry["strength"]))
                    for entry in job.get("loras", [])
                ]
                resolved_jobs.append({**job, "loras": loras})
            run_batch_on_pipe(
                pipe,
                resolved_jobs,
                width=width,
                height=height,
                default_steps=DEFAULT_COMPARE_STEPS,
            )
        finally:
            release_text_encoder(pipe)
    else:
        generate_batch_via_worker(
            jobs=jobs,
            width=width,
            height=height,
            steps=DEFAULT_COMPARE_STEPS,
            precision=precision,
        )

    return outputs
