"""Cache management CLI."""
from __future__ import annotations

import json
import sys

from .config import apply_env
from .prompt_embed_cache import list_entries, prune, remove_entry


def run_cache(argv: list[str]) -> int:
    apply_env()
    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: cli.py cache {list|prune|rm HASH}")
        return 0

    cmd = argv[0]
    if cmd == "list":
        entries = list_entries()
        if not entries:
            print("cache empty")
            return 0
        for entry in entries:
            prompt = entry.get("prompt", "")
            preview = prompt[:60] + ("…" if len(prompt) > 60 else "")
            print(
                f"{entry.get('hash', '?')[:12]}… "
                f"hits={entry.get('hits', 0)} "
                f"encode={entry.get('encode_time_s', '?')}s "
                f"size={entry.get('size_bytes', 0)} "
                f"last={entry.get('last_used_at', '?')} "
                f"{preview!r}"
            )
        print(f"{len(entries)} entries", file=sys.stderr)
        return 0

    if cmd == "prune":
        stats = prune()
        print(json.dumps(stats))
        return 0

    if cmd == "rm":
        if len(argv) < 2:
            print("cache rm needs a hash prefix or full hash", file=sys.stderr)
            return 1
        needle = argv[1]
        matches = [e for e in list_entries() if e.get("hash", "").startswith(needle)]
        if not matches:
            print(f"no cache entry matching {needle!r}", file=sys.stderr)
            return 1
        for entry in matches:
            remove_entry(entry["hash"])
            print(f"removed {entry['hash']}")
        return 0

    print(f"unknown cache command: {cmd}", file=sys.stderr)
    return 1
