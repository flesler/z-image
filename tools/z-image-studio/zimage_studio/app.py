from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import run_benchmark
from .cache_cmd import run_cache
from .compare import collect_prompts, parse_steps_list, run_compare
from .config import apply_env
from .daemon import main as daemon_main
from .generate import run_generate


def _add_gen_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lora", action="append", default=[], metavar="NAME[:STRENGTH]")
    p.add_argument("--precision", default="q4")
    p.add_argument("--width", "-w", type=int, default=1024)
    p.add_argument("--height", "-H", type=int, default=1024)
    p.add_argument("--seed", type=int)
    p.add_argument("--steps", type=int, default=9)
    p.add_argument("--output", "-o", type=Path)
    p.add_argument("--cold", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("gen", help="generate a single image")
    gen.add_argument("prompt")
    _add_gen_flags(gen)

    compare = sub.add_parser("compare", help="A/B compare base vs LoRA(s)")
    compare.add_argument("--prompt", action="append", default=None, metavar="TEXT", help="repeatable; seeds use --seed, --seed+1, ...")
    compare.add_argument("--prompt-file", action="append", default=[], type=Path, metavar="FILE", help="one prompt per line; skips empty and duplicates")
    compare.add_argument("--lora", action="append", required=True, metavar="NAME[:STRENGTH]|*", help="repeatable; use '*' for all catalog LoRAs")
    compare.add_argument("--seed", type=int)
    compare.add_argument("--repeat", type=int, default=1)
    compare.add_argument("--width", "-w", type=int, default=1024)
    compare.add_argument("--height", "-H", type=int, default=1024)
    compare.add_argument("--steps", default="9", metavar="N[,N...]", help="inference steps, or comma-separated (e.g. 7,8,9,10)")
    compare.add_argument("--precision", default="q4")
    compare.add_argument("--each", action="store_true")
    compare.add_argument("--combo", action="store_true")
    compare.add_argument("--cold", action="store_true")
    compare.add_argument("--override", action="store_true")

    bench = sub.add_parser("benchmark", help="run catalog LoRA compares")
    bench.add_argument("--lora", action="append", default=[], metavar="NAME[:STRENGTH]")
    bench.add_argument("--seed-base", type=int, default=401)
    bench.add_argument("--repeat", type=int, default=1)
    bench.add_argument("--cold", action="store_true")
    bench.add_argument("--override", action="store_true")

    daemon = sub.add_parser("daemon", help="warm worker daemon")
    daemon.add_argument("action", choices=["start", "stop", "status", "logs"])

    cache = sub.add_parser("cache", help="prompt embed disk cache")
    cache.add_argument("action", nargs="?", default="list", choices=["list", "prune", "rm"])
    cache.add_argument("hash", nargs="?", help="hash prefix for rm")

    return parser


def main(argv: list[str] | None = None) -> int:
    apply_env()
    args = build_parser().parse_args(argv)

    if args.command == "gen":
        run_generate(
            args.prompt,
            loras=args.lora,
            precision=args.precision,
            width=args.width,
            height=args.height,
            seed=args.seed,
            steps=args.steps,
            output=args.output,
            cold=args.cold,
        )
        return 0

    if args.command == "compare":
        run_compare(
            collect_prompts(inline=args.prompt, files=args.prompt_file),
            loras=args.lora,
            seed=args.seed,
            seed_set=args.seed is not None,
            repeat=args.repeat,
            width=args.width,
            height=args.height,
            steps_list=parse_steps_list(args.steps),
            precision=args.precision,
            each=args.each,
            combo=args.combo,
            cold=args.cold,
            override=args.override,
        )
        return 0

    if args.command == "benchmark":
        run_benchmark(
            filters=args.lora or None,
            seed_base=args.seed_base,
            repeat=args.repeat,
            cold=args.cold,
            override=args.override,
        )
        return 0

    if args.command == "daemon":
        return daemon_main([args.action])

    if args.command == "cache":
        argv = [args.action]
        if args.hash:
            argv.append(args.hash)
        return run_cache(argv)

    return 1
