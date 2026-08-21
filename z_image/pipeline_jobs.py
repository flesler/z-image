"""Shared image generation helpers for worker and cold path."""
from __future__ import annotations

import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import torch
from diffusers import ZImageImg2ImgPipeline
from safetensors.torch import load_file
from PIL import Image

from .config import default_precision, load_pipeline
from .exceptions import ClientDisconnected
from .metadata import GenMeta, save_image
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


def recover_pipe(pipe) -> None:
    """Reset pipeline state after errors or client disconnect."""
    if pipe is None:
        return
    try:
        pipe.transformer.unload_lora()
    except Exception:
        pass
    release_text_encoder(pipe)
    scheduler = getattr(pipe, "scheduler", None)
    if scheduler is not None:
        if hasattr(scheduler, "set_begin_index"):
            try:
                scheduler.set_begin_index(0)
            except Exception:
                pass
        if hasattr(scheduler, "_step_index"):
            scheduler._step_index = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _lora_key(loras: list[tuple[str, float]] | None) -> tuple[tuple[str, float], ...]:
    if not loras:
        return ()
    return tuple(loras)


def _job_group_key(job: dict) -> tuple:
    loras = job.get("loras") or []
    if loras and isinstance(loras[0], dict):
        lora_key = tuple((entry["file"], float(entry.get("strength", 1.0))) for entry in loras)
    else:
        lora_key = _lora_key(loras or None)
    return (job["prompt"], int(job["seed"]), lora_key)


def _collapse_step_jobs(model_jobs: list[dict], job_steps: Callable[[dict], int]) -> list[dict]:
    """Merge pending jobs that share prompt/seed/lora; run once at max remaining steps."""
    collapsed: list[dict] = []
    i = 0
    while i < len(model_jobs):
        key = _job_group_key(model_jobs[i])
        group = [model_jobs[i]]
        i += 1
        while i < len(model_jobs) and _job_group_key(model_jobs[i]) == key:
            group.append(model_jobs[i])
            i += 1
        if len(group) == 1:
            collapsed.append(group[0])
            continue
        snapshots = {job_steps(entry): entry["output"] for entry in group}
        collapsed.append(
            {
                "prompt": group[0]["prompt"],
                "seed": group[0]["seed"],
                "loras": group[0].get("loras"),
                "steps": max(snapshots),
                "snapshots": snapshots,
            }
        )
    return collapsed


def _latents_to_pil(pipe, latents):
    latents = latents.to(pipe.vae.dtype)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


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


def _img2img_on_pipe(
    pipe,
    *,
    steps: int,
    width: int,
    height: int,
    seed: int,
    init_image: Image.Image,
    strength: float,
    prompt: str | None = None,
    prompt_embeds: list[torch.Tensor] | None = None,
):
    if (prompt is None) == (prompt_embeds is None):
        raise ValueError("provide exactly one of prompt or prompt_embeds")

    img2img = ZImageImg2ImgPipeline.from_pipe(pipe, torch_dtype=None)
    generator = torch.Generator(device=pipe_device(pipe)).manual_seed(seed)
    gen_kwargs = {
        "image": init_image,
        "strength": strength,
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
        return img2img(**gen_kwargs).images[0]


def _denoise_on_pipe_with_snapshots(
    pipe,
    *,
    max_steps: int,
    snapshots: dict[int, str | Path],
    width: int,
    height: int,
    seed: int,
    prompt_embeds: list[torch.Tensor],
    on_snapshot: Callable[[dict], None] | None = None,
    loras_count: int = 0,
    meta: GenMeta | None = None,
) -> None:
    """One max-step denoise; decode predicted x0 at each pending step (not noisy latents)."""
    from diffusers.pipelines.z_image.pipeline_z_image import (
        calculate_shift,
        get_default_z_image_sigmas,
        retrieve_timesteps,
    )

    pending = set(snapshots.keys())
    t_last = time.perf_counter()
    device = pipe_device(pipe)
    generator = torch.Generator(device=device).manual_seed(seed)
    prompt_embeds = [tensor.to(device) for tensor in prompt_embeds]

    with torch.inference_mode():
        latents = pipe.prepare_latents(
            1,
            pipe.transformer.in_channels,
            height,
            width,
            torch.float32,
            device,
            generator,
            None,
        )
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        sigmas = get_default_z_image_sigmas(max_steps)
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler,
            max_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        pipe.scheduler.set_begin_index(0)

        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000
            latent_model_input = latents.to(pipe.transformer.dtype).unsqueeze(2)
            latent_model_input_list = list(latent_model_input.unbind(dim=0))
            model_out_list = pipe.transformer(
                latent_model_input_list,
                timestep,
                prompt_embeds,
                return_dict=False,
            )[0]
            noise_pred = torch.stack([out.float() for out in model_out_list], dim=0).squeeze(2)
            # Z-Image convention: negate before scheduler.step
            noise_pred = -noise_pred

            if pipe.scheduler.step_index is None:
                pipe.scheduler._init_step_index(t)
            sigma = pipe.scheduler.sigmas[pipe.scheduler.step_index]
            # Flow-match predicted clean latent (VAE-decodable), not noisy x_t
            x0 = latents.float() - sigma * noise_pred

            completed = i + 1
            if completed in pending:
                image = _latents_to_pil(pipe, x0)
                output = Path(snapshots[completed])
                if meta is not None:
                    save_image(
                        image,
                        output,
                        GenMeta(
                            prompt=meta.prompt,
                            width=meta.width,
                            height=meta.height,
                            seed=meta.seed,
                            steps=completed,
                            precision=meta.precision,
                            loras=meta.loras,
                            strength=meta.strength,
                        ),
                    )
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    image.save(output)
                pending.discard(completed)
                if on_snapshot:
                    now = time.perf_counter()
                    on_snapshot(
                        {
                            "output": str(output),
                            "loras": loras_count,
                            "steps": completed,
                            "elapsed_s": round(now - t_last, 2),
                        }
                    )
                    t_last = now

            latents = pipe.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]

    if pending:
        raise RuntimeError(f"step snapshots missing after denoise: {sorted(pending)}")


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
    init_image: Image.Image | None = None,
    strength: float | None = None,
):
    _load_loras(pipe, loras)
    try:
        if init_image is not None:
            if strength is None:
                raise ValueError("strength is required when init_image is set")
            return _img2img_on_pipe(
                pipe,
                steps=steps,
                width=width,
                height=height,
                seed=seed,
                init_image=init_image,
                strength=strength,
                prompt=prompt,
                prompt_embeds=prompt_embeds,
            )
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
    reuse_steps: bool = False,
    init_image: Image.Image | None = None,
    strength: float | None = None,
    log: Callable[[str], None] | None = None,
    reloaded: bool = False,
    on_image: Callable[[dict], None] | None = None,
) -> tuple[list[dict], float, float]:
    """Run jobs grouped by model, then steps, then prompts."""
    if not jobs:
        return [], 0.0, 0.0
    if init_image is not None and reuse_steps:
        raise ValueError("img2img batch does not support --reuse-steps")

    prepare_loaded_pipe(pipe, reloaded=reloaded)
    if torch.cuda.is_available():
        dev = pipe_device(pipe)
        if dev.type == "cuda":
            torch.cuda.set_device(dev)

    def job_steps(job: dict) -> int:
        return int(job["steps"])

    emit = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    model_groups: OrderedDict[tuple[tuple[str, float], ...], list[dict]] = OrderedDict()
    for job in jobs:
        key = _lora_key(job.get("loras") or None)
        model_groups.setdefault(key, []).append(job)

    step_values = sorted({job_steps(job) for job in jobs})
    mode = "img2img" if init_image is not None else "txt2img"
    emit(
        f"batch: {len(jobs)} image(s) across {len(model_groups)} model(s), "
        f"{len({job['prompt'] for job in jobs})} unique prompt(s), {mode}"
        + (f", steps {','.join(str(s) for s in step_values)}" if len(step_values) > 1 else "")
        + (f", strength {strength:g}" if init_image is not None and strength is not None else "")
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
    aborted = False

    for loras_key, model_jobs in model_groups.items():
        if aborted:
            break
        loras = list(loras_key) if loras_key else None
        label = "base" if not loras else f"lora×{len(loras)}"
        multi_steps = len({job_steps(job) for job in model_jobs}) > 1
        collapsed_jobs = (
            _collapse_step_jobs(model_jobs, job_steps) if reuse_steps else model_jobs
        )
        emit(f"model {label}: {len(model_jobs)} image(s)")
        _load_loras(pipe, loras)
        try:
            for job in collapsed_jobs:
                if aborted:
                    break
                snapshots: dict[int, str | Path] | None = job.get("snapshots")
                if snapshots:
                    def on_snapshot(result: dict) -> None:
                        nonlocal img_n, aborted
                        if aborted:
                            return
                        img_n += 1
                        emit(
                            f"image {img_n}/{len(jobs)}: {result['elapsed_s']:.1f}s "
                            f"{label} s{result['steps']} → {Path(result['output']).name}"
                        )
                        results.append(result)
                        if on_image:
                            try:
                                on_image(result)
                            except ClientDisconnected:
                                aborted = True
                                raise

                    try:
                        job_meta = GenMeta(
                            prompt=job["prompt"],
                            width=width,
                            height=height,
                            seed=int(job["seed"]),
                            steps=job_steps(job),
                            precision=default_precision(),
                            loras=loras,
                            strength=strength,
                        )
                        _denoise_on_pipe_with_snapshots(
                            pipe,
                            max_steps=job_steps(job),
                            snapshots=snapshots,
                            width=width,
                            height=height,
                            seed=int(job["seed"]),
                            prompt_embeds=embeds_by_prompt[job["prompt"]],
                            on_snapshot=on_snapshot,
                            loras_count=len(loras or []),
                            meta=job_meta,
                        )
                    except ClientDisconnected:
                        aborted = True
                        break
                    continue

                step_count = job_steps(job)
                step_tag = f" s{step_count}" if multi_steps else ""
                img_n += 1
                t_img = time.perf_counter()
                if init_image is not None:
                    if strength is None:
                        raise ValueError("strength is required for img2img batch")
                    image = _img2img_on_pipe(
                        pipe,
                        steps=step_count,
                        width=width,
                        height=height,
                        seed=int(job["seed"]),
                        init_image=init_image,
                        strength=strength,
                        prompt_embeds=embeds_by_prompt[job["prompt"]],
                    )
                else:
                    image = _denoise_on_pipe(
                        pipe,
                        steps=step_count,
                        width=width,
                        height=height,
                        seed=int(job["seed"]),
                        prompt_embeds=embeds_by_prompt[job["prompt"]],
                    )
                output = Path(job["output"])
                save_image(
                    image,
                    output,
                    GenMeta(
                        prompt=job["prompt"],
                        width=width,
                        height=height,
                        seed=int(job["seed"]),
                        steps=step_count,
                        precision=default_precision(),
                        loras=loras,
                        strength=strength,
                    ),
                )
                img_s = time.perf_counter() - t_img
                emit(f"image {img_n}/{len(jobs)}: {img_s:.1f}s {label}{step_tag} → {output.name}")
                result = {
                    "output": str(output),
                    "loras": len(loras or []),
                    "steps": step_count,
                    "elapsed_s": round(img_s, 2),
                }
                results.append(result)
                if on_image:
                    try:
                        on_image(result)
                    except ClientDisconnected:
                        aborted = True
                        break
        except ClientDisconnected:
            aborted = True
        finally:
            _unload_loras(pipe, loras)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if aborted:
        emit("batch aborted (client disconnected)")
        raise ClientDisconnected("client disconnected")

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
    init_image: Image.Image | None = None,
    strength: float | None = None,
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
            init_image=init_image,
            strength=strength,
        )
    except Exception:
        traceback.print_exc()
        raise
    finally:
        recover_pipe(pipe)
