# Source before running zimg: source tools/z-image-studio/.env.sh
export Z_IMAGE_STUDIO_DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"
export Z_IMAGE_STUDIO_OUTPUT_DIR="$Z_IMAGE_STUDIO_DATA_DIR/outputs"
export HF_HOME="$Z_IMAGE_STUDIO_DATA_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv/bin:$PATH"
