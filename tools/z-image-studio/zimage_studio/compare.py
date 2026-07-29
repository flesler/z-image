from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .config import apply_env, compare_dir
from .generate import run_generate
from .loras import LoraSpec, random_seed, resolve_spec
from .naming import compare_filename, compare_stem, slugify
from .worker_client import ensure_worker


def run_compare(
    prompt: str,
    *,
    loras: list[str],
    seed: int | None = None,
    seed_set: bool = False,
    repeat: int = 1,
    width: int = 1024,
    height: int = 1024,
    steps: int = 9,
    precision: str = "q4",
    each: bool = False,
    combo: bool = False,
    cold: bool = False,
    override: bool = False,
) -> list[Path]:
    apply_env()
    if not loras:
        raise SystemExit("need at least one --lora name[:strength]")
    if repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    resolved = [resolve_spec(spec) for spec in loras]
    if not each and not combo:
        each = True
        combo = len(resolved) > 1

    root = compare_dir()
    root.mkdir(parents=True, exist_ok=True)
    prompt_slug = slugify(prompt)

    if not cold:
        ensure_worker(cold=False)

    outputs: list[Path] = []

    for rep in range(repeat):
        if repeat > 1:
            print(f"repeat {rep + 1}/{repeat}", file=sys.stderr)
        if seed_set:
            iter_seed = (seed or 0) + rep
        else:
            iter_seed = random_seed()
        outputs.extend(
            _run_one_compare(
                prompt=prompt,
                resolved=resolved,
                iter_seed=iter_seed,
                root=root,
                width=width,
                height=height,
                steps=steps,
                precision=precision,
                each=each,
                combo=combo,
                cold=cold,
                override=override,
            )
        )

    print(f"compare root: {root}", file=sys.stderr)
    pattern = f"{prompt_slug}--{width}x{height}--s*"
    listed = sorted(root.glob(pattern))
    for path in listed:
        print(path)
    return listed


def _run_one_compare(
    *,
    prompt: str,
    resolved: list[LoraSpec],
    iter_seed: int,
    root: Path,
    width: int,
    height: int,
    steps: int,
    precision: str,
    each: bool,
    combo: bool,
    cold: bool,
    override: bool,
) -> list[Path]:
    stem = compare_stem(prompt, width, height, iter_seed)
    print(f"seed: {iter_seed}", file=sys.stderr)
    shared_base: Path | None = None
    outputs: list[Path] = []

    def run_gen(label: str, output: Path, lora_specs: list[str] | None = None) -> None:
        if not override and output.is_file():
            print(f"⊘ skip {label} (exists) → {output}", file=sys.stderr)
            return
        print(f"→ {label} → {output}", file=sys.stderr)
        run_generate(
            prompt,
            loras=lora_specs or [],
            seed=iter_seed,
            width=width,
            height=height,
            steps=steps,
            precision=precision,
            output=output,
            cold=cold,
        )
        outputs.append(output)

    def ensure_base() -> Path:
        nonlocal shared_base
        base_out = root / compare_filename(stem, "base")
        if not override and base_out.is_file():
            shared_base = base_out
            print(f"⊘ skip base (exists) → {base_out}", file=sys.stderr)
            return base_out
        if shared_base and shared_base.is_file():
            if shared_base != base_out:
                shutil.copy2(shared_base, base_out)
                print(f"→ base (cached) → {base_out}", file=sys.stderr)
            return base_out
        run_gen("base", base_out)
        shared_base = base_out
        return base_out

    def compare_pair(spec: LoraSpec) -> None:
        lora_out = root / compare_filename(stem, spec.name)
        ensure_base()
        run_gen(f"lora {spec}", lora_out, [str(spec)])

    if each:
        for spec in resolved:
            compare_pair(spec)

    if combo and len(resolved) > 1:
        combo_name = "_combo" + "".join(f"+{spec.name}" for spec in resolved)
        lora_out = root / compare_filename(stem, combo_name)
        ensure_base()
        run_gen(f"lora combo ({len(resolved)})", lora_out, [str(spec) for spec in resolved])
    elif combo and len(resolved) == 1 and not each:
        compare_pair(resolved[0])

    return outputs
