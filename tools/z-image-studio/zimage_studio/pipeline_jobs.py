"""Shared image generation helpers for worker and cold path."""
from __future__ import annotations

import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import torch
from safetensors.torch import load_file

from .config import load_pipeline
from .text_encoder import encode_prompts, pipe_device, release_text_encoder


def _load_loras(pipe, loras: list[tuple[str, float]] | None) -> None:
    if not loras:
        return
    active = []
    weights = []
    for i, (path, strength) in enumerate(loras):
        adapter_name = f"lora_{i}"
        state_dict = load_file(path)
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("diffusion_model."):
                remapped[key.replace("diffusion_model.", "transformer.")] = value
            else:
                remapped[key] = value
        pipe.transformer.load_lora_adapter(
            remapped,
            adapter_name=adapter_name,
            prefix="transformer",
        )
        active.append(adapter_name)
        weights.append(strength)
    if active:
        pipe.transformer.set_adapters(active, weights=weights)


def _unload_loras(pipe, loras: list[tuple[str, float]] | None) -> None:
    if not loras:
        return
    try:
        pipe.transformer.unload_lora()
    except Exception as e:
        print(f"[worker] failed to unload LoRA: {e}", flush=True)


def _lora_key(loras: list[tuple[str, float]] | None) -> tuple[tuple[str, float], ...]:
    if not loras:
        return ()
    return tuple(loras)


def _denoise_on_pipe(
    pipe,
    *,
    steps: int,
    width: int,
    height: int,
    seed: int,
    prompt: str | None = None,
    prompt_embeds: list[torch.Tensor] | None = None,
):
    if (prompt is None) == (prompt_embeds is None):
        raise ValueError("provide exactly one of prompt or prompt_embeds")

    generator = torch.Generator(device=pipe_device(pipe)).manual_seed(seed)
    gen_kwargs = {
        "num_inference_steps": steps,
        "height": height,
        "width": width,
        "guidance_scale": 0.0,
        "generator": generator,
    }
    if prompt_embeds is not None:
        device = next(pipe.transformer.parameters()).device
        prompt_embeds = [tensor.to(device) for tensor in prompt_embeds]
        gen_kwargs["prompt"] = None
        gen_kwargs["prompt_embeds"] = prompt_embeds
    else:
        gen_kwargs["prompt"] = prompt

    with torch.inference_mode():
        return pipe(**gen_kwargs).images[0]


def run_on_pipe(
    pipe,
    *,
    steps: int,
    width: int,
    height: int,
    seed: int,
    loras: list[tuple[str, float]] | None,
    prompt: str | None = None,
    prompt_embeds: list[torch.Tensor] | None = None,
):
    _load_loras(pipe, loras)
    try:
        return _denoise_on_pipe(
            pipe,
            steps=steps,
            width=width,
            height=height,
            seed=seed,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
        )
    finally:
        _unload_loras(pipe, loras)


def prepare_loaded_pipe(pipe, *, reloaded: bool = False) -> None:
    """Re-apply fast-path settings; full .to(cuda) only needed after idle reload."""
    from .config import cpu_offload_enabled

    if reloaded and torch.cuda.is_available() and not cpu_offload_enabled():
        pipe.to("cuda")
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "disable_attention_slicing"):
        pipe.disable_attention_slicing()
    release_text_encoder(pipe)


def run_batch_on_pipe(
    pipe,
    jobs: list[dict],
    *,
    width: int,
    height: int,
    default_steps: int = 9,
    log: Callable[[str], None] | None = None,
    reloaded: bool = False,
) -> tuple[list[dict], float, float]:
    """Run jobs grouped by model, then steps, then prompts."""
    if not jobs:
        return [], 0.0, 0.0

    prepare_loaded_pipe(pipe, reloaded=reloaded)
    if torch.cuda.is_available():
        dev = pipe_device(pipe)
        if dev.type == "cuda":
            torch.cuda.set_device(dev)

    def job_steps(job: dict) -> int:
        return int(job.get("steps", default_steps))

    emit = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    model_groups: OrderedDict[tuple[tuple[str, float], ...], list[dict]] = OrderedDict()
    for job in jobs:
        key = _lora_key(job.get("loras") or None)
        model_groups.setdefault(key, []).append(job)

    step_values = sorted({job_steps(job) for job in jobs})
    emit(
        f"batch: {len(jobs)} image(s) across {len(model_groups)} model(s), "
        f"{len({job['prompt'] for job in jobs})} unique prompt(s)"
        + (f", steps {','.join(str(s) for s in step_values)}" if len(step_values) > 1 else "")
    )

    all_prompts = list(dict.fromkeys(job["prompt"] for job in jobs))
    t0 = time.perf_counter()
    emit(f"encode {len(all_prompts)} unique prompt(s)")
    t_enc = time.perf_counter()
    embeds_by_prompt = encode_prompts(pipe, all_prompts)
    encode_s = time.perf_counter() - t_enc
    emit(f"encode phase: {encode_s:.1f}s")

    results: list[dict] = []
    img_n = 0

    for loras_key, model_jobs in model_groups.items():
        loras = list(loras_key) if loras_key else None
        label = "base" if not loras else f"lora×{len(loras)}"
        steps_groups: OrderedDict[int, list[dict]] = OrderedDict()
        for job in model_jobs:
            steps_groups.setdefault(job_steps(job), []).append(job)
        emit(f"model {label}: {len(model_jobs)} image(s)")
        _load_loras(pipe, loras)
        try:
            for step_count, step_jobs in steps_groups.items():
                step_tag = f" s{step_count}" if len(steps_groups) > 1 else ""
                for job in step_jobs:
                    img_n += 1
                    t_img = time.perf_counter()
                    image = _denoise_on_pipe(
                        pipe,
                        steps=step_count,
                        width=width,
                        height=height,
                        seed=int(job["seed"]),
                        prompt_embeds=embeds_by_prompt[job["prompt"]],
                    )
                    output = Path(job["output"])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    image.save(output)
                    img_s = time.perf_counter() - t_img
                    emit(f"image {img_n}/{len(jobs)}: {img_s:.1f}s {label}{step_tag} → {output.name}")
                    results.append(
                        {
                            "output": str(output),
                            "loras": len(loras or []),
                            "steps": step_count,
                            "elapsed_s": round(img_s, 2),
                        }
                    )
        finally:
            _unload_loras(pipe, loras)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    denoise_s = time.perf_counter() - t0 - encode_s
    return results, encode_s, denoise_s


def generate_one(
    *,
    precision: str,
    steps: int,
    width: int,
    height: int,
    seed: int,
    loras: list[tuple[str, float]] | None,
    prompt: str | None = None,
    prompt_embeds: list[torch.Tensor] | None = None,
):
    pipe = load_pipeline(precision=precision)
    if prompt is not None and prompt_embeds is None:
        prompt_embeds = encode_prompts(pipe, [prompt])[prompt]
        prompt = None
    try:
        return run_on_pipe(
            pipe,
            steps=steps,
            width=width,
            height=height,
            seed=seed,
            loras=loras,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
        )
    except Exception:
        traceback.print_exc()
        raise
    finally:
        release_text_encoder(pipe)
