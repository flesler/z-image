"""Load, encode, and fully release the text encoder."""
from __future__ import annotations

import gc
import sys
import time
from typing import Iterable

import torch
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from zimage.hardware import MODEL_ID_MAP

from .config import cpu_offload_enabled
from . import prompt_embed_cache as embed_cache


def _log(msg: str) -> None:
    print(f"[gpu] {msg}", file=sys.stderr, flush=True)


def pipe_device(pipe: ZImagePipeline) -> torch.device:
    return next(pipe.transformer.parameters()).device


def release_text_encoder(pipe: ZImagePipeline) -> None:
    if cpu_offload_enabled() or pipe.text_encoder is None:
        return
    _log("text encoder released")
    del pipe.text_encoder
    pipe.text_encoder = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_text_encoder(pipe: ZImagePipeline) -> None:
    if cpu_offload_enabled() or pipe.text_encoder is not None:
        return

    precision = getattr(pipe, "_zimage_precision", "q4")
    model_id = MODEL_ID_MAP[precision]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    import sdnq  # noqa: F401 — register SDNQ quantizers
    from sdnq.common import use_torch_compile as triton_is_available
    from sdnq.loader import apply_sdnq_options_to_model
    from transformers import AutoModelForCausalLM

    _log("loading text encoder")
    text_encoder = AutoModelForCausalLM.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if triton_is_available and torch.cuda.is_available():
        text_encoder = apply_sdnq_options_to_model(text_encoder, use_quantized_matmul=True)
    pipe.text_encoder = text_encoder.to("cuda")


def encode_prompts(pipe: ZImagePipeline, prompts: Iterable[str]) -> dict[str, list[torch.Tensor]]:
    unique = list(dict.fromkeys(prompts))
    if not unique:
        return {}

    precision = getattr(pipe, "_zimage_precision", "q4")
    device = pipe_device(pipe)
    t0 = time.perf_counter()
    cached: dict[str, list[torch.Tensor]] = {}
    to_encode: list[str] = []

    for prompt in unique:
        hit = embed_cache.load(prompt, precision)
        if hit is not None:
            cached[prompt] = embed_cache.to_device(hit, device)
        else:
            to_encode.append(prompt)

    if to_encode:
        ensure_text_encoder(pipe)
        for prompt in to_encode:
            t_one = time.perf_counter()
            embeds, _ = pipe.encode_prompt(prompt, do_classifier_free_guidance=False)
            embed_cache.save(prompt, precision, embeds, encode_time_s=time.perf_counter() - t_one)
            cached[prompt] = embeds
        release_text_encoder(pipe)
        _log(
            f"encoded {len(to_encode)} prompt(s), {len(unique) - len(to_encode)} disk hit(s) "
            f"in {time.perf_counter() - t0:.1f}s"
        )
    else:
        if pipe.text_encoder is not None:
            _log("text encoder resident despite cache hits — releasing")
            release_text_encoder(pipe)
        _log(
            f"all {len(unique)} prompt(s) from disk cache — text encoder not loaded "
            f"({time.perf_counter() - t0:.2f}s)"
        )

    return cached
