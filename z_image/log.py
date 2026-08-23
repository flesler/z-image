"""ISO-timestamped stderr logging."""
from __future__ import annotations

import sys
from datetime import datetime

_out = sys.stderr


def log(msg: str, *, tag: str | None = None, file=None) -> None:
    ts = datetime.now().astimezone().isoformat(timespec="milliseconds")
    prefix = f"{ts} "
    if tag:
        prefix += f"[{tag}] "
    print(f"{prefix}{msg}", file=file or _out, flush=True)
