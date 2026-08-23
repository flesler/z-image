"""Fast GPU path for 8GB cards — no accelerate sequential offload."""
from __future__ import annotations

import os
import time
from typing import Callable, Type

import torch
import zimage.engine as engine
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from diffusers.pipelines.z_image.pipeline_z_image_img2img import ZImageImg2ImgPipeline

from .config import cpu_offload_enabled, gpu_monitor_enabled, resolve_steps, verbose_enabled
from .log import log as tslog
from .text_encoder import ensure_text_encoder, release_text_encoder

_installed = False


def vae_tiling_enabled() -> bool:
    return os.environ.get("Z_IMAGE_VAE_TILING", "0").lower() in ("1", "true", "yes")


def _patch_pipeline_class(
    cls: Type,
    *,
    patch_prepare_latents: bool,
) -> None:
    if getattr(cls, "_zimage_gpu_patched", False):
        return

    original_call = cls.__call__
    original_prepare_latents = getattr(cls, "prepare_latents", None)

    def __call__(self, *args, **kwargs):
        if not cpu_offload_enabled() and kwargs.get("prompt") is not None:
            ensure_text_encoder(self)
        if gpu_monitor_enabled() and not cpu_offload_enabled():
            steps = resolve_steps(kwargs.get("num_inference_steps"))
            t0 = time.perf_counter()

            def on_step(p, step_index, _timestep, callback_kwargs):
                if step_index in (0, steps - 1):
                    tfm = next(p.transformer.parameters()).device
                    te = "released"
                    if p.text_encoder is not None:
                        te = str(next(p.text_encoder.parameters()).device)
                    elapsed = time.perf_counter() - t0
                    log(
                        f"step {step_index + 1}/{steps} "
                        f"tfm={tfm} te={te} elapsed={elapsed:.1f}s",
                        tag="gpu",
                    )
                return callback_kwargs

            if kwargs.get("callback_on_step_end") is None:
                kwargs["callback_on_step_end"] = on_step
                kwargs.setdefault("callback_on_step_end_tensor_inputs", ["latents"])
        result = original_call(self, *args, **kwargs)
        if not cpu_offload_enabled() and not patch_prepare_latents:
            release_text_encoder(self)
        return result

    cls.__call__ = __call__
    if patch_prepare_latents and original_prepare_latents is not None:

        def prepare_latents(self, *args, **kwargs):
            if not cpu_offload_enabled():
                release_text_encoder(self)
            return original_prepare_latents(self, *args, **kwargs)

        cls.prepare_latents = prepare_latents
    cls._zimage_gpu_patched = True


def _patch_pipeline() -> None:
    _patch_pipeline_class(ZImagePipeline, patch_prepare_latents=True)
    _patch_pipeline_class(ZImageImg2ImgPipeline, patch_prepare_latents=False)


def install(*, log: Callable[[str], None] | None = None) -> None:
    global _installed
    if _installed:
        return

    def emit(msg: str) -> None:
        if log:
            log(msg)
        else:
            tslog(msg, tag="gpu")

    if cpu_offload_enabled():
        if verbose_enabled():
            emit("accelerate CPU offload (Z_IMAGE_CPU_OFFLOAD=1)")
        _installed = True
        return

    original = engine.load_pipeline
    real_offload = ZImagePipeline.enable_model_cpu_offload

    def load_pipeline_fast_gpu(device=None, precision="q4"):
        cached = getattr(engine, "_cached_pipe", None)
        if cached is not None:
            release_text_encoder(cached)

        def noop_offload(self, *args, **kwargs):
            return self

        ZImagePipeline.enable_model_cpu_offload = noop_offload
        try:
            pipe = original(device=device, precision=precision)
        finally:
            ZImagePipeline.enable_model_cpu_offload = real_offload

        pipe._zimage_precision = precision
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        if hasattr(pipe, "disable_attention_slicing"):
            pipe.disable_attention_slicing()
        if torch.cuda.is_available():
            if vae_tiling_enabled() and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            torch.cuda.empty_cache()
        release_text_encoder(pipe)
        return pipe

    engine.load_pipeline = load_pipeline_fast_gpu
    _patch_pipeline()
    if verbose_enabled():
        tiling = ", VAE tiling on" if vae_tiling_enabled() else ""
        emit(f"fast path — no attention slicing, no progress bar, TE released after encode{tiling}")
    _installed = True
