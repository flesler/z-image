# Model

Steps: 8–9 (rarely more than 10–12; extra steps usually add little).
CFG / guidance_scale: 0.0–1.0 (official examples often use 0.0 or ~1.0). Higher values frequently cause oversaturation, plastic skin, or warped details.
Negative prompts: Mostly ignored or ineffective on Turbo. Put constraints directly in the positive prompt instead (e.g., “correct anatomy, five natural fingers, no extra limbs, no distorted hands”).
Resolution: Keep total pixels near native (~1024×1024) and dimensions divisible by 16 or 32. Extreme aspect ratios or oversized canvases increase face/hand distortion risk. Upscale afterward if needed.

Use natural-language, descriptive prose (not SD-style tag soup). Structure roughly as: subject + appearance/action (include hands/face details early) + clothing + environment + lighting + camera/style + cleanup constraints.

Helpful phrases:
- Hands: “well-formed hands, natural five fingers, detailed knuckles and nails, correct anatomy, relaxed natural pose”
- Faces: “detailed realistic face, natural skin texture with pores, subtle imperfections, sharp focused features, natural expression”
- General: “correct anatomy, sharp focus, highly detailed skin”

Describe the specific action involving the hands (e.g., “gently holding a glass orb with both hands, fingers carefully curled around it”) rather than just hoping for good hands. Photographic language helps realism: “shot on 85mm lens, soft natural window light, shallow depth of field, film grain.” The base model is already relatively strong on faces (especially East Asian) and better than older models on many poses, but explicit description + good seed still helps a lot.

# LoRA batch

```bash
ZIMG=./cli.py

$ZIMG daemon start
$ZIMG gen "prompt" --lora DarkGhibliZ
$ZIMG batch "prompt" --lora DarkGhibliZ
$ZIMG benchmark
```

Same prompt, seed, and dimensions — base vs LoRA(s) side by side:

Omit `--seed` for a random seed (printed to stderr). Pass `--seed N` to reproduce.
Use `--repeat N` to run N variants; each repeat gets a new seed (random, or `N..N+repeat-1` when `--seed` is set). Base and all LoRAs share the seed within each repeat.

```bash
$ZIMG batch "your prompt here" \
  --lora RealisticSnapshot-Zimage-Turbov5 \
  --repeat 5 --each
```

Omit `:strength` to use `default_strength` from `lib/loras.json`. Batch all catalog prompts:

```bash
$ZIMG benchmark
```

Outputs land flat in `/tmp/z-image/batch/` — sort by filename to group variants for the same prompt/seed:

```
/tmp/z-image/batch/
  <prompt-slug>-1024x1024-s77-base.png
  <prompt-slug>-1024x1024-s77-DarkGhibliZ.png
  <prompt-slug>-1024x1024-s77-ZiTMythG0thicL1nes.png
```

Same stem, different suffix — scales to A/B/C/D without per-LoRA folders.

Existing outputs are skipped by default; pass `--override` to regenerate. Augment a set with one more LoRA:

```bash
$ZIMG batch "same prompt" \
  --lora ZiTMythG0thicL1nes \
  --lora ZiTMythR3alisticF \
  --lora Purple_grainy_zit \
  --seed 301 --each
```

Only `Purple_grainy_zit` is generated if base and the other two already exist.

Flags: `--each` (per-LoRA files), `--combo` (all LoRAs stacked, when 2+). Default runs both when multiple LoRAs are passed.

Trigger words for known LoRAs are prepended automatically from `lib/loras.json` when `--lora` is used (base/batch control image is unchanged).

Single image with LoRA: `$ZIMG gen "prompt" --lora DarkGhibliZ`

# Reverse caption

Extract a natural-language prompt from an image, or generate variants without writing a prompt:

```bash
$ZIMG caption photo.png
$ZIMG from-image photo.png --repeat 3
$ZIMG from-image photo.png --caption-only   # print prompt only
```

Uses BLIP-large in the warm worker (`daemon start`). `--caption-device auto` (default) picks CPU when the gen model is on GPU; otherwise GPU fp16 when VRAM allows. Caption model unloads on idle timeout (`Z_IMAGE_IDLE_UNLOAD_MINUTES`, same as generation).

Benchmark helpers (optional): `scripts/caption_eval.py`, `scripts/caption_benchmark.py`.

