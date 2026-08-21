# z-image

CLI and long-running worker for [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) image generation with LoRA support, batch comparison, and img2img.

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Install torch (CUDA) and zimage per your environment, then:
pip install -e .

source .env.sh
./cli.py daemon start
./cli.py gen "your prompt"
```

See [tips.md](tips.md) for model guidance and batch workflows. See [docs/randomness.md](docs/randomness.md) for seed diversity options.

## Caption / from-image

Reverse-caption an image (BLIP-large, warm in the worker like generation):

```bash
./cli.py caption path/to/image.png
./cli.py from-image path/to/image.png --repeat 3
```

`--caption-device auto|cpu|gpu` (default `auto`): GPU when enough free VRAM; CPU when the gen pipeline is loaded. First caption after idle loads the model (~10s); repeat calls are fast until `Z_IMAGE_IDLE_UNLOAD_MINUTES` evicts it.
