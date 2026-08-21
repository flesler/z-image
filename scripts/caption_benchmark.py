#!/usr/bin/env python3
"""Round-trip caption benchmark: source → caption → regenerate → CLIP similarity."""
from __future__ import annotations

import gc
import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from z_image.caption import caption_image, unload_caption_model

EVAL_DIR = ROOT / "data" / "outputs" / "caption-benchmark"
CLI = ROOT / "cli.py"
CLIP_ID = "openai/clip-vit-large-patch14"
REGEN_SEED = 6000

PROMPTS = [
    "candid street portrait of a woman laughing, amateur snapshot, urban background, natural imperfect framing",
    "a lonely child walking through an enchanted forest at dusk, fireflies, mossy ancient trees, soft mist",
    "cozy witch cottage interior, warm candlelight, stacked books, sleeping cats, wooden beams",
    "neon-lit alley at night, lonely figure walking away, cinematic grain, moody purple haze",
]

CAPTION_MODELS = [
    {
        "id": "blip-large",
        "model": "Salesforce/blip-image-captioning-large",
        "prompt": "a detailed photograph showing",
    },
    {
        "id": "blip-base",
        "model": "Salesforce/blip-image-captioning-base",
        "prompt": "a detailed photograph showing",
    },
    {
        "id": "blip-large-uncond",
        "model": "Salesforce/blip-image-captioning-large",
        "prompt": "",
    },
    {
        "id": "blip2-opt-cpu",
        "model": "blip2-opt-2.7b-cpu",
        "prompt": "Question: Describe this image in detail for an AI image generator. Answer:",
    },
]


def run_cli(*args: str) -> None:
    subprocess.run([str(CLI), *args], check=True)


def free_gpu() -> None:
    unload_caption_model()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def clip_similarity(a: Path, b: Path, clip_model, clip_proc, device: str) -> float:
    images = [Image.open(p).convert("RGB") for p in (a, b)]
    inputs = clip_proc(images=images, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        out = clip_model.get_image_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return float((feats[0] @ feats[1]).item())


def caption_blip2_cpu(path: Path, prompt: str) -> str:
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32
    ).to("cpu")
    image = Image.open(path).convert("RGB")
    inputs = processor(image, text=prompt, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=120, num_beams=1)
    text = processor.decode(out[0], skip_special_tokens=True)
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1].strip()
    del model, processor
    gc.collect()
    return text


def extract(path: Path, spec: dict) -> str:
    if spec["model"] == "blip2-opt-2.7b-cpu":
        return caption_blip2_cpu(path, spec["prompt"])
    if spec["prompt"]:
        return caption_image(path, model=spec["model"], prompt=spec["prompt"])
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(spec["model"])
    model = BlipForConditionalGeneration.from_pretrained(spec["model"]).to("cuda")
    image = Image.open(path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=80, num_beams=5)
    text = processor.decode(out[0], skip_special_tokens=True).strip()
    del model, processor
    free_gpu()
    return text


def extract_all(sources: list[dict]) -> dict[str, list[dict]]:
    captions_path = EVAL_DIR / "captions.json"
    if captions_path.is_file():
        return json.loads(captions_path.read_text())

    run_cli("daemon", "stop")
    free_gpu()

    all_caps: dict[str, list[dict]] = {}
    for spec in CAPTION_MODELS:
        print(f"extracting with {spec['id']}...", file=sys.stderr)
        rows = []
        for src in sources:
            extracted = extract(Path(src["path"]), spec)
            free_gpu()
            rows.append({"source_id": src["id"], "original_prompt": src["prompt"], "extracted": extracted})
            print(f"  src{src['id']}: {extracted[:100]}...", file=sys.stderr)
        all_caps[spec["id"]] = rows

    captions_path.write_text(json.dumps(all_caps, indent=2) + "\n")
    free_gpu()
    return all_caps


def regenerate_all(captions: dict[str, list[dict]]) -> None:
    run_cli("daemon", "start")
    for model_id, rows in captions.items():
        for row in rows:
            out = EVAL_DIR / f"roundtrip-{model_id}-src{row['source_id']}.png"
            if out.is_file():
                continue
            print(f"regen {model_id} src{row['source_id']}...", file=sys.stderr)
            run_cli("gen", row["extracted"], "--seed", str(REGEN_SEED), "-o", str(out))


def score_all(captions: dict[str, list[dict]], sources: list[dict]) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_proc = CLIPProcessor.from_pretrained(CLIP_ID)
    clip_model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
    src_by_id = {s["id"]: Path(s["path"]) for s in sources}

    results = []
    for model_id, rows in captions.items():
        sims = []
        samples = []
        for row in rows:
            src_path = src_by_id[row["source_id"]]
            rt_path = EVAL_DIR / f"roundtrip-{model_id}-src{row['source_id']}.png"
            sim = clip_similarity(src_path, rt_path, clip_model, clip_proc, device)
            sims.append(sim)
            samples.append({**row, "roundtrip": str(rt_path), "clip_sim": round(sim, 4)})
        avg = sum(sims) / len(sims)
        results.append({"model": model_id, "avg_clip_sim": round(avg, 4), "samples": samples})
    return results


def main() -> int:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    sources: list[dict] = []
    for i, prompt in enumerate(PROMPTS, 1):
        seed = 5000 + i
        out = EVAL_DIR / f"source-{i}-s{seed}.png"
        if not out.is_file():
            run_cli("daemon", "start")
            run_cli("gen", prompt, "--seed", str(seed), "-o", str(out))
        sources.append({"id": i, "prompt": prompt, "seed": seed, "path": str(out)})

    captions = extract_all(sources)
    regenerate_all(captions)
    results = score_all(captions, sources)

    out = EVAL_DIR / "benchmark-results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")

    ranked = sorted(results, key=lambda r: r["avg_clip_sim"], reverse=True)
    print("\nRANKING (avg CLIP sim source→roundtrip):", file=sys.stderr)
    for r in ranked:
        print(f"  {r['model']}: {r['avg_clip_sim']:.4f}", file=sys.stderr)
    print(f"\n→ {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
