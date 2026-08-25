"""Resolution-based VAE decode VRAM tactics for 8GB GPUs."""
from __future__ import annotations

import torch
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline

from .config import cpu_offload_enabled

# 1024² fits transformer+VAE decode; ~1080×1400+ needs transformer parked (~3GB headroom).
VAE_PARK_PIXELS = 1024 * 1400
# Tiled decode is slower — only when output is taller than ~1088×1920 reel.
VAE_TILE_PIXELS = 1024 * 2048


def decode_output_pixels(latents: torch.Tensor, pipe: ZImagePipeline) -> int:
    if not isinstance(latents, torch.Tensor) or latents.ndim < 4:
        return 0
    scale = int(getattr(pipe, "vae_scale_factor", 8))
    h, w = int(latents.shape[-2]), int(latents.shape[-1])
    return h * scale * w * scale


def should_park_transformer_for_decode(pixels: int) -> bool:
    if cpu_offload_enabled() or not torch.cuda.is_available() or pixels <= 0:
        return False
    return pixels >= VAE_PARK_PIXELS


def should_tile_vae_for_decode(pixels: int) -> bool:
    if pixels <= 0:
        return False
    return pixels >= VAE_TILE_PIXELS
