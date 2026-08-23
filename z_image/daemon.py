from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .config import PIDFILE, ROOT, VENV_BIN, apply_env, default_precision, worker_log, worker_port
from .log import log as tslog
from .worker_client import fetch_health, is_healthy, is_ready


def cmd_start() -> int:
    apply_env()
    logfile = worker_log()
    logfile.parent.mkdir(parents=True, exist_ok=True)

    if is_ready():
        print(f"worker already running on :{worker_port()}")
        return 0

    if PIDFILE.is_file():
        old_pid = int(PIDFILE.read_text().strip())
        try:
            os.kill(old_pid, 0)
        except OSError:
            PIDFILE.unlink(missing_ok=True)
        else:
            tslog(f"worker pid {old_pid} exists but health check failed")
            return 1

    python = VENV_BIN / "python"
    logfh = logfile.open("a", buffering=1)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [str(python), "-m", "z_image.worker_server", default_precision()],
        cwd=ROOT,
        stdout=logfh,
        stderr=subprocess.STDOUT,
        env=env,
    )
    # Keep log handle open for the worker process lifetime (do not close here).
    PIDFILE.write_text(str(proc.pid))
    print(f"starting worker (pid {proc.pid}) — tail {logfile}")

    for _ in range(60):
        if is_healthy():
            state = "ready" if is_ready() else f"warming {default_precision()}"
            print(f"worker listening on :{worker_port()} ({state})")
            return 0
        time.sleep(0.5)

    tslog("worker failed to start within 30s")
    return 1


def cmd_stop() -> int:
    if not PIDFILE.is_file():
        print("worker not running")
        return 0
    pid = int(PIDFILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"stopped worker pid {pid}")
    except OSError:
        print(f"stale pidfile (pid {pid} not running)")
    PIDFILE.unlink(missing_ok=True)
    return 0


def cmd_restart() -> int:
    cmd_stop()
    time.sleep(1)
    return cmd_start()


def cmd_status() -> int:
    health = fetch_health()
    if health:
        pid = PIDFILE.read_text().strip() if PIDFILE.is_file() else "?"
        state = health.get("status", "unknown")
        print(f"worker on :{worker_port()} (pid {pid}) — {state}")
        return 0
    print("worker not running")
    return 1


def cmd_logs() -> int:
    logfile = worker_log()
    subprocess.run(["tail", "-f", str(logfile)], check=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: z-image daemon {start|stop|restart|status|logs}")
        return 0 if not argv else 1

    cmd = argv[0]
    if cmd == "start":
        return cmd_start()
    if cmd == "stop":
        return cmd_stop()
    if cmd == "restart":
        return cmd_restart()
    if cmd == "status":
        return cmd_status()
    if cmd == "logs":
        return cmd_logs()
    tslog(f"unknown daemon command: {cmd}")
    return 1
