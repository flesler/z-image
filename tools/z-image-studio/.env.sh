# Source before running zimg: source tools/z-image-studio/.env.sh
export Z_IMAGE_STUDIO_DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"
export Z_IMAGE_STUDIO_OUTPUT_DIR="$Z_IMAGE_STUDIO_DATA_DIR/outputs"
export HF_HOME="$Z_IMAGE_STUDIO_DATA_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ZIMAGE_CPU_OFFLOAD="${ZIMAGE_CPU_OFFLOAD:-0}"   # 1 = slow accelerate offload fallback
export ZIMAGE_VAE_TILING="${ZIMAGE_VAE_TILING:-0}"     # 1 = enable only if decode OOMs
export ZIMAGE_GPU_MONITOR="${ZIMAGE_GPU_MONITOR:-1}"   # log device placement per denoise step
export ZIMAGE_PROMPT_EMBED_CACHE="${ZIMAGE_PROMPT_EMBED_CACHE:-1}"  # disk cache for prompt embeds
export ZIMAGE_IDLE_UNLOAD_MINUTES="${ZIMAGE_IDLE_UNLOAD_MINUTES:-5}"  # 0 = keep model loaded forever
export PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv/bin:$PATH"
