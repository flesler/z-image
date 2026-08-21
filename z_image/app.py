from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .benchmark import run_benchmark
from .cache_cmd import run_cache
from .batch import collect_prompts, dedupe_ints, normalize_steps_list, run_batch
from .config import DEFAULT_IMG2IMG_STRENGTH, DEFAULT_STEPS, PREVIEW_STEPS, apply_env
from .caption import run_caption
from .daemon import main as daemon_main
from .from_image import run_from_image
from .generate import run_generate
from .sizes import ASPECT_BASE_CHOICES, resolve_dimensions, size_choices


def _add_size_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--size",
        choices=size_choices(),
        metavar="PRESET",
        help="preset dimensions: " + ", ".join(size_choices()),
    )
    p.add_argument(
        "--aspect-ratio",
        metavar="W:H",
        help="width:height ratio; width from --aspect-base (default 1024, use 1080 for social)",
    )
    p.add_argument(
        "--aspect-base",
        type=int,
        choices=list(ASPECT_BASE_CHOICES),
        default=1024,
        help="reference width for --aspect-ratio (1080 snaps to 1088 for VAE)",
    )
    p.add_argument("--width", "-w", type=int, default=None)
    p.add_argument("--height", "-H", type=int, default=None)


def _add_caption_device_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--caption-device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="reverse-caption device: auto picks GPU when enough free VRAM, else CPU",
    )


def _add_gen_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lora", action="append", default=[], metavar="NAME[:STRENGTH]")
    p.add_argument("--precision", default=None)
    _add_size_flags(p)
    p.add_argument("--seed", type=int)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--image", type=Path, metavar="PATH", help="init image for img2img")
    p.add_argument(
        "--strength",
        type=float,
        default=None,
        metavar="S",
        help=f"img2img denoise strength 0-1 (default {DEFAULT_IMG2IMG_STRENGTH})",
    )
    p.add_argument("--output", "-o", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("gen", help="generate a single image")
    gen.add_argument("prompt")
    _add_gen_flags(gen)

    caption = sub.add_parser("caption", help="reverse-caption an image to stdout")
    caption.add_argument("image", type=Path)
    _add_caption_device_flag(caption)

    from_img = sub.add_parser(
        "from-image",
        help="caption an image and generate variant(s) at the source resolution",
    )
    from_img.add_argument("image", type=Path, help="source image to reverse-caption")
    from_img.add_argument(
        "--caption-only",
        action="store_true",
        help="print extracted prompt only; do not generate",
    )
    from_img.add_argument("--lora", action="append", default=[], metavar="NAME[:STRENGTH]")
    from_img.add_argument("--precision", default=None)
    _add_size_flags(from_img)
    from_img.add_argument("--seed", action="append", type=int, default=None, metavar="N")
    from_img.add_argument("--repeat", type=int, default=1, help="variant count when no --seed")
    from_img.add_argument("--steps", type=int, default=None)
    from_img.add_argument("--output", "-o", type=Path)
    _add_caption_device_flag(from_img)

    batch = sub.add_parser("batch", help="batch generate prompts across base and LoRA(s)")
    batch.add_argument("--prompt", action="append", default=None, metavar="TEXT", help="repeatable")
    batch.add_argument("--prompt-file", action="append", default=[], type=Path, metavar="FILE", help="one prompt per line; skips empty and duplicates")
    batch.add_argument("--lora", action="append", default=None, metavar="NAME[:STRENGTH]|*", help="repeatable; omit for base-only; use '*' for all catalog LoRAs")
    batch.add_argument("--seed", action="append", type=int, default=None, metavar="N", help="repeatable base seed; with --repeat N uses N+1 seeds: base..base+N")
    batch.add_argument("--repeat", type=int, default=1, help="extra seeds after base: --repeat 1 → 2 seeds, --repeat 3 → 4 seeds")
    _add_size_flags(batch)
    batch.add_argument("--steps", action="append", type=int, default=None, metavar="N", help=f"repeatable inference steps; default {DEFAULT_STEPS}")
    batch.add_argument(
        "--preview",
        action="store_true",
        help=f"fast preview at {PREVIEW_STEPS} steps; same filenames as default (no -s{{N}} suffix)",
    )
    batch.add_argument("--precision", default=None)
    batch.add_argument("--image", type=Path, metavar="PATH", help="shared init image for img2img batch")
    batch.add_argument(
        "--strength",
        type=float,
        default=None,
        metavar="S",
        help=f"img2img denoise strength 0-1 (default {DEFAULT_IMG2IMG_STRENGTH})",
    )
    batch.add_argument("--no-base", action="store_true", help="skip base model images (default: include base)")
    batch.add_argument("--each", action="store_true")
    batch.add_argument("--combo", action="store_true")
    batch.add_argument(
        "--reuse-steps",
        action="store_true",
        help="merge same prompt/seed/model runs: one denoise at max steps, partial x0 snapshots (faster, lower quality)",
    )
    batch.add_argument("--override", action="store_true")
    batch.add_argument("--dry-run", action="store_true", help="print planned outputs only, do not generate")
    batch.add_argument("--verbose", action="store_true", help="extra diagnostics (implied by --dry-run)")

    bench = sub.add_parser("benchmark", help="run catalog LoRA batches")
    bench.add_argument("--lora", action="append", default=[], metavar="NAME[:STRENGTH]")
    bench.add_argument("--seed-base", type=int, default=401)
    bench.add_argument("--repeat", type=int, default=1)
    bench.add_argument("--override", action="store_true")

    daemon = sub.add_parser("daemon", help="warm worker daemon")
    daemon.add_argument("action", choices=["start", "stop", "restart", "status", "logs"])

    cache = sub.add_parser("cache", help="prompt embed disk cache")
    cache.add_argument("action", nargs="?", default="list", choices=["list", "prune", "rm"])
    cache.add_argument("hash", nargs="?", help="hash prefix for rm")

    return parser


def _log_cli_total(t0: float) -> None:
    elapsed = time.perf_counter() - t0
    print(f"total {elapsed:.2f}s", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None, *, t0: float | None = None) -> int:
    started = t0 if t0 is not None else time.perf_counter()
    args = build_parser().parse_args(argv)
    if getattr(args, "verbose", False) or getattr(args, "dry_run", False):
        os.environ["Z_IMAGE_VERBOSE"] = "1"
    apply_env()

    if args.command == "gen":
        if args.strength is not None and args.image is None:
            raise SystemExit("--strength requires --image")
        width, height = resolve_dimensions(
            size=args.size,
            aspect_ratio=args.aspect_ratio,
            aspect_base=args.aspect_base,
            width=args.width,
            height=args.height,
        )
        run_generate(
            args.prompt,
            loras=args.lora,
            precision=args.precision,
            width=width,
            height=height,
            seed=args.seed,
            steps=args.steps,
            output=args.output,
            image=args.image,
            strength=args.strength,
        )
        _log_cli_total(started)
        return 0

    if args.command == "caption":
        print(run_caption(args.image, device=args.caption_device))
        _log_cli_total(started)
        return 0

    if args.command == "from-image":
        if args.size or args.aspect_ratio:
            raise SystemExit("--size and --aspect-ratio are ignored; use --width/--height or source image size")
        if (args.width is None) != (args.height is None):
            raise SystemExit("provide both --width and --height, or omit both to use source image size")
        run_from_image(
            args.image,
            loras=args.lora,
            precision=args.precision,
            width=args.width,
            height=args.height,
            seed=args.seed[0] if args.seed and len(args.seed) == 1 and args.repeat == 1 else None,
            seeds=args.seed,
            repeat=args.repeat,
            steps=args.steps,
            output=args.output,
            caption_only=args.caption_only,
            caption_device=args.caption_device,
        )
        _log_cli_total(started)
        return 0

    if args.command == "batch":
        if args.preview and args.steps:
            raise SystemExit("--preview and --steps are mutually exclusive")
        seeds = dedupe_ints(args.seed)
        width, height = resolve_dimensions(
            size=args.size,
            aspect_ratio=args.aspect_ratio,
            aspect_base=args.aspect_base,
            width=args.width,
            height=args.height,
        )
        run_batch(
            collect_prompts(inline=args.prompt, files=args.prompt_file),
            loras=args.lora or [],
            seeds=seeds,
            seed_set=bool(seeds),
            repeat=args.repeat,
            width=width,
            height=height,
            steps_list=normalize_steps_list(args.steps),
            preview=args.preview,
            precision=args.precision,
            each=args.each,
            combo=args.combo,
            include_base=not args.no_base,
            reuse_steps=args.reuse_steps,
            override=args.override,
            dry_run=args.dry_run,
            image=args.image,
            strength=args.strength,
        )
        _log_cli_total(started)
        return 0

    if args.command == "benchmark":
        run_benchmark(
            filters=args.lora or None,
            seed_base=args.seed_base,
            repeat=args.repeat,
            override=args.override,
        )
        _log_cli_total(started)
        return 0

    if args.command == "daemon":
        return daemon_main([args.action])

    if args.command == "cache":
        argv = [args.action]
        if args.hash:
            argv.append(args.hash)
        return run_cache(argv)

    return 1
