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

# LoRA A/B compare

```bash
ZIMG=tools/z-image-studio/cli.py

$ZIMG daemon start
$ZIMG gen "prompt" --lora DarkGhibliZ
$ZIMG compare "prompt" --lora DarkGhibliZ
$ZIMG benchmark
```

Same prompt, seed, and dimensions — base vs LoRA(s) side by side:

Omit `--seed` for a random seed (printed to stderr). Pass `--seed N` to reproduce.
Use `--repeat N` to run N variants; each repeat gets a new seed (random, or `N..N+repeat-1` when `--seed` is set). Base and all LoRAs share the seed within each repeat.

```bash
$ZIMG compare "your prompt here" \
  --lora RealisticSnapshot-Zimage-Turbov5 \
  --repeat 5 --each
```

Omit `:strength` to use `default_strength` from `lib/loras.json`. Batch all catalog prompts:

```bash
$ZIMG benchmark
```

Outputs land flat in `/tmp/z-image/compare/` — sort by filename to group variants for the same prompt/seed:

```
/tmp/z-image/compare/
  <prompt-slug>--1024x1024--s77-base.png
  <prompt-slug>--1024x1024--s77-DarkGhibliZ.png
  <prompt-slug>--1024x1024--s77-ZiTMythG0thicL1nes.png
```

Same stem, different suffix — scales to A/B/C/D without per-LoRA folders.

Existing outputs are skipped by default; pass `--override` to regenerate. Augment a set with one more LoRA:

```bash
$ZIMG compare "same prompt" \
  --lora ZiTMythG0thicL1nes \
  --lora ZiTMythR3alisticF \
  --lora Purple_grainy_zit \
  --seed 301 --each
```

Only `Purple_grainy_zit` is generated if base and the other two already exist.

Flags: `--each` (per-LoRA files), `--combo` (all LoRAs stacked, when 2+). Default runs both when multiple LoRAs are passed.

Trigger words for known LoRAs are prepended automatically from `lib/loras.json` when `--lora` is used (base/compare control image is unchanged).

Single image with LoRA: `$ZIMG gen "prompt" --lora DarkGhibliZ`

# LoRAs

## DarkGhibliZ

https://civitai.com/models/1349631/dark-ghibli-fairytales?modelVersionId=2500043

Trigger (auto-prepended): `Studio Ghibli Dark Fairytale` — default strength 0.85 in `loras.json`.

## Purple_grainy_zit

https://civitai.com/models/2329053/purple-grainy-oror-photography-lora?modelVersionId=2619939

Default strength 1.0 — grainy photography style (no trigger word).

## ZiTMythG0thicL1nes

https://civitai.com/models/599757/velvets-mythic-fantasy-styles-or-flux-pony-illustrious-zit-anima-krea2?modelVersionId=2924569

Trigger (auto-prepended): `G0thicL1nes` — default strength 0.8, mythic fantasy line-art style.

## ZiTMythR3alisticF

https://civitai.com/models/599757/velvets-mythic-fantasy-styles-or-flux-pony-illustrious-zit-anima-krea2?modelVersionId=2547883

Trigger (auto-prepended): `R3alisticF` — default strength 0.8, mythic fantasy painterly realism.

## MidJourneyNSFWZ

https://civitai.com/models/837884/midjourney-artful-nsfw?modelVersionId=2599899

Trigger (auto-prepended): `ArtfulNSFW` — default strength 0.8, MidJourney-style art.

## RealisticSnapshot

Default strength 0.65 in `loras.json` (potent; override lower if needed).

Trigger words (optional — push texture/lighting style):

- Camera roll look: amateur digital snapshot, candid, smartphone capture, high ISO noise, direct on-camera flash
- High-fidelity details: visible pores, visible vellus hair, subsurface scattering, detailed skin texture
- Optical realism: wide-angle lens, barrel distortion, chromatic aberration, depth of field
