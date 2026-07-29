from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .config import LORAS_JSON, loras_dir


@dataclass(frozen=True)
class LoraSpec:
    file: str
    strength: float

    @property
    def name(self) -> str:
        return self.file.removesuffix(".safetensors")

    def __str__(self) -> str:
        return f"{self.file}:{self.strength}"


def random_seed() -> int:
    return random.randint(1, 2**31 - 1)


def normalize_filename(name: str) -> str:
    base = Path(name).name
    if not base.endswith(".safetensors"):
        base += ".safetensors"
    return base


def load_catalog(path: Path | None = None) -> dict:
    catalog_path = path or LORAS_JSON
    if not catalog_path.is_file():
        return {}
    with catalog_path.open(encoding="utf-8") as f:
        return json.load(f)


def catalog_entry(catalog: dict, name: str) -> dict:
    file = normalize_filename(name)
    return catalog.get(file) or catalog.get(file.removesuffix(".safetensors"), {})


def resolve_spec(spec: str, catalog: dict | None = None) -> LoraSpec:
    catalog = catalog if catalog is not None else load_catalog()
    file, _, strength_str = spec.partition(":")
    file = normalize_filename(file)
    default = float(catalog_entry(catalog, file).get("default_strength", 1.0))
    strength = float(strength_str) if strength_str else default
    return LoraSpec(file=file, strength=strength)


def resolve_path(file: str, root: Path | None = None) -> Path:
    root = root or loras_dir()
    normalized = normalize_filename(file)
    for candidate in (Path(file), root / normalized):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"LoRA not found: {file}")


def apply_triggers(prompt: str, specs: list[str], catalog: dict | None = None) -> str:
    if not specs:
        return prompt
    catalog = catalog if catalog is not None else load_catalog()
    seen: list[str] = []
    pl = prompt.lower()
    for spec in specs:
        entry = catalog_entry(catalog, spec.split(":", 1)[0])
        trigger = (entry.get("trigger") or "").strip()
        if trigger and trigger.lower() not in pl and trigger not in seen:
            seen.append(trigger)
            pl += " " + trigger.lower()
    if seen:
        return ", ".join(seen) + ", " + prompt
    return prompt


def filter_names(filters: list[str]) -> set[str]:
    names: set[str] = set()
    for spec in filters:
        file = normalize_filename(spec.split(":", 1)[0])
        names.add(file)
        names.add(file.removesuffix(".safetensors"))
    return names


def benchmark_plan(
    catalog: dict,
    seed_base: int,
    repeat: int,
    filters: list[str] | None = None,
) -> list[tuple[str, str, int]]:
    names = filter_names(filters or [])
    rows: list[tuple[str, str, int]] = []
    idx = 0
    for file, entry in catalog.items():
        if names and Path(file).name not in names and file not in names:
            continue
        prompts = entry.get("prompts") or []
        if not prompts:
            continue
        for prompt in prompts:
            rows.append((file, prompt, seed_base + idx * repeat))
            idx += 1
    return rows
