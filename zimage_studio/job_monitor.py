"""CPU and fan sampling during worker jobs."""
from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _worker_pid() -> int:
    return os.getpid()


def _read_fan_rpms() -> list[int]:
    rpms: list[int] = []
    for path in Path("/sys/class/hwmon").glob("hwmon*/fan*_input"):
        try:
            value = int(path.read_text().strip())
            if value > 0:
                rpms.append(value)
        except (OSError, ValueError):
            continue
    return rpms


def _read_gpu_temp_c() -> float | None:
    for path in Path("/sys/class/hwmon").glob("hwmon*/temp*_input"):
        name_file = path.with_name(path.name.replace("temp", "name").replace("_input", ""))
        try:
            label = name_file.read_text().strip().lower() if name_file.is_file() else ""
        except OSError:
            label = ""
        if "gpu" in label or "nvidia" in label or "nouveau" in label:
            try:
                milli = int(path.read_text().strip())
                return milli / 1000.0
            except (OSError, ValueError):
                continue
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2,
        ).strip()
        return float(out.splitlines()[0])
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        return None


class JobMonitor:
    def __init__(self, *, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu_by_tid: dict[str, dict[str, float]] = defaultdict(lambda: {"max": 0.0, "sum": 0.0, "n": 0, "core": -1, "comm": ""})
        self._proc_cpu_max = 0.0
        self._fan_max = 0
        self._fan_last: list[int] = []
        self._gpu_temp_max: float | None = None
        self._samples = 0

    def __enter__(self) -> "JobMonitor":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        pid = _worker_pid()
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["ps", "-L", "-p", str(pid), "-o", "tid,psr,pcpu,comm"],
                    text=True,
                    timeout=1,
                )
                for line in out.strip().splitlines()[1:]:
                    parts = line.split(None, 3)
                    if len(parts) < 4:
                        continue
                    tid, core, pcpu, comm = parts[0], int(parts[1]), float(parts[2]), parts[3]
                    slot = self._cpu_by_tid[tid]
                    slot["max"] = max(slot["max"], pcpu)
                    slot["sum"] += pcpu
                    slot["n"] += 1
                    slot["core"] = core
                    slot["comm"] = comm
                    if tid == str(pid):
                        self._proc_cpu_max = max(self._proc_cpu_max, pcpu)
            except (subprocess.SubprocessError, ValueError):
                pass

            fans = _read_fan_rpms()
            if fans:
                self._fan_last = fans
                self._fan_max = max(self._fan_max, max(fans))

            temp = _read_gpu_temp_c()
            if temp is not None:
                if self._gpu_temp_max is None:
                    self._gpu_temp_max = temp
                else:
                    self._gpu_temp_max = max(self._gpu_temp_max, temp)

            self._samples += 1
            time.sleep(self.interval_s)

    def summary(self) -> dict[str, Any]:
        top_tid = None
        top = 0.0
        top_core = -1
        for tid, slot in self._cpu_by_tid.items():
            if slot["max"] > top:
                top = slot["max"]
                top_tid = tid
                top_core = int(slot["core"])

        return {
            "samples": self._samples,
            "proc_cpu_max_pct": round(self._proc_cpu_max, 1),
            "hot_thread_max_pct": round(top, 1),
            "hot_thread_tid": top_tid,
            "hot_thread_core": top_core,
            "fan_rpm_last": self._fan_last,
            "fan_rpm_max": self._fan_max or None,
            "gpu_temp_c_max": round(self._gpu_temp_max, 1) if self._gpu_temp_max is not None else None,
        }


def log_summary(summary: dict[str, Any], *, label: str) -> None:
    fan = summary.get("fan_rpm_max")
    fan_s = f" fan_max={fan}rpm" if fan else " fan=n/a"
    temp = summary.get("gpu_temp_c_max")
    temp_s = f" gpu_temp_max={temp}C" if temp is not None else ""
    print(
        f"[monitor] {label}: proc_cpu_max={summary['proc_cpu_max_pct']}% "
        f"hot_thread={summary['hot_thread_max_pct']}%@core{summary['hot_thread_core']}"
        f"{fan_s}{temp_s} samples={summary['samples']}",
        flush=True,
    )
