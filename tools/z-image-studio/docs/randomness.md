# Seed diversity & controlled randomness

Z-Image-Turbo is fast and prompt-faithful, but **different seeds often produce surprisingly similar images** — same pose, face, composition, only minor detail shifts. This is a known Turbo tradeoff from distillation, not a bug in this tool.

This doc lists ways to get meaningful variation, what exists today, and what we plan to add.

---

## What the tool does today

| Mechanism | CLI | Effect on Turbo diversity |
|-----------|-----|---------------------------|
| `--seed N` | gen, batch | Weak — reproducible, but outputs often look alike |
| `--repeat N` | batch | Runs seeds `base..base+N`; same limitation |
| Prompt edits | manual | **Strong** — Turbo is very text-driven |
| `--image` + `--strength` | gen, batch | **Strong** — img2img from a shared init image |
| CFG / negative prompts | — | Not useful on Turbo (`guidance_scale=0` is correct) |

Generation uses a single `torch.Generator` seeded from `--seed` for initial latents. No extra noise injection or embedding perturbation yet.

---

## Options to add (planned)

Three complementary approaches, all deterministic when keyed to seed (reproducible runs).

### 1. Latent jitter (RandomNoise-like)

Perturb **initial latents** before denoising:

```
latents = prepare_latents(...)
latents = latents + jitter_scale * noise_from(noise_seed)
```

| | |
|---|---|
| **Proposed flags** | `--latent-jitter`, `--noise-seed` (optional, default derived from seed) |
| **Speed** | ~0 — one tensor add before the denoise loop |
| **Diversity on Turbo** | Weak–moderate — model was distilled to be robust to latent noise |
| **Prompt adherence** | Usually preserved |
| **Failure mode** | Often barely changes; very high jitter → mushy starts |
| **Implementation** | Easy — diffusers already accepts `latents=` on `__call__` |

### 2. Embedding variance (SeedVarianceEnhancer-like)

Perturb **text embeddings** during early denoise steps, then switch back to clean embeds (ComfyUI [SeedVarianceEnhancer](https://github.com/ChangeTheConstants/SeedVarianceEnhancer) approach).

| | |
|---|---|
| **Proposed flags** | `--embed-variance`, `--embed-variance-pct`, `--embed-variance-steps` (names TBD) |
| **Speed** | ~0 — no extra forward passes; swap conditioning inside existing loop |
| **Diversity on Turbo** | Moderate–strong — community-proven for this model |
| **Prompt adherence** | **Degrades** as strength rises |
| **Failure mode** | Anatomy glitches, extra limbs, worse text-in-image at high settings |
| **Starting point** | strength ~15–30, ~50% of dims, noise on **beginning steps only** (Comfy defaults) |
| **Implementation** | Medium — extend `pipeline_jobs` denoise loop (similar to reuse-steps path) |

### 3. Deterministic prompt suffix

Append something derived from seed to the prompt before encode:

```
"{prompt}, {suffix(seed)}"
```

Two flavours:

| Suffix style | Diversity | Notes |
|--------------|-----------|-------|
| **Hash fragment** (`v7f3a2`) | Weak–unpredictable | Easy; model may ignore junk tokens |
| **Curated phrase bank** (`three-quarter view`, `warmer light`, …) indexed by `seed % N` | Moderate–strong | Best effort/safety ratio for Turbo |

| | |
|---|---|
| **Proposed flags** | `--seed-suffix` or `--vary-prompt` (mode: `hash` \| `phrases`) |
| **Speed** | +text encode per unique suffix on first run (cached after); batch with many seeds = N encodes |
| **Diversity on Turbo** | Moderate–strong if phrases are meaningful |
| **Prompt adherence** | Controlled by phrase list — drift is interpretable |
| **Failure mode** | Gibberish suffix ignored; concrete nouns may appear in rendered text |
| **Implementation** | Easy — append before `encode_prompts`; new cache key per suffix |

---

## Comparison

| | Latent jitter | Embed variance | Prompt suffix |
|--|---------------|----------------|---------------|
| Speed (steady state) | ~0 | ~0 | ~0 (+ encode on cache miss) |
| Turbo diversity | ★★☆ | ★★★★ | ★★★–★★★★ (phrases) |
| Prompt control | ★★★★★ | ★★☆ | ★★★★ (if curated) |
| Anatomy safety | ★★★★★ | ★★☆ | ★★★★ |
| Reproducibility | yes | yes | yes |
| Proven on Turbo | less | yes (Comfy) | anecdotal |

**Combining:** suffix + light embed variance is possible but compounds adherence loss. Prefer tuning one knob at a time.

---

## Workarounds available now (no new flags)

### img2img chain

Best in-repo option for controlled composition change:

```bash
cli.py gen "blonde elf portrait …" --seed 42 -o /tmp/elf.png
cli.py gen "Santa Claus, red suit …" --image /tmp/elf.png --strength 0.65 --seed 42 -o /tmp/santa.png
```

`--strength` 0.55–0.75: lower keeps pose/framing; higher rewrites more.

### Manual prompt variation

Small wording changes often beat seed sweeps on Turbo. Vary angle, lighting, expression, environment — not just `--repeat`.

### Seed sweeps

```bash
cli.py batch --prompt "…" --seed 100 --repeat 9
```

Cheap to try; often disappointing for face/pose diversity alone.

### ComfyUI / external

SeedVarianceEnhancer and similar nodes live in Comfy today if you need them before we ship CLI flags.

---

## Recommended rollout order

1. **Deterministic prompt suffix** (curated phrase bank) — easy, Turbo-native, predictable tradeoffs  
2. **Embedding variance** — when suffix alone isn’t enough  
3. **Latent jitter** — optional extra knob; low cost, often weak alone  

Defaults: all off (`0` / disabled). No behavior change unless flags are set.

---

## References

- [SeedVarianceEnhancer](https://github.com/ChangeTheConstants/SeedVarianceEnhancer) — ComfyUI node, embedding noise for Turbo  
- [Apatero guide](https://apatero.com/blog/seedvarianceenhancer-z-image-diversity-complete-guide-2025) — settings and tradeoffs  
- In-repo: `tips.md` (model behaviour), img2img via `--image` / `--strength`
