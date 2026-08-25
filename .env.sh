# Source before running zimg: source .env.sh
export Z_IMAGE_DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"
export Z_IMAGE_DATA_OUTPUT_DIR="$Z_IMAGE_DATA_DIR/outputs"
export HF_HOME="$Z_IMAGE_DATA_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export Z_IMAGE_CPU_OFFLOAD="${Z_IMAGE_CPU_OFFLOAD:-0}"   # 1 = slow accelerate offload fallback
export Z_IMAGE_GPU_MONITOR="${Z_IMAGE_GPU_MONITOR:-1}"   # log device placement per denoise step
export Z_IMAGE_PROMPT_EMBED_CACHE="${Z_IMAGE_PROMPT_EMBED_CACHE:-1}"  # disk cache for prompt embeds
export Z_IMAGE_IDLE_UNLOAD_MINUTES="${Z_IMAGE_IDLE_UNLOAD_MINUTES:-5}"  # 0 = keep model loaded forever
export Z_IMAGE_WORKER_HOST="${Z_IMAGE_WORKER_HOST:-0.0.0.0}"
export Z_IMAGE_WORKER_PORT="${Z_IMAGE_WORKER_PORT:-8000}"
export PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv/bin:$PATH"
