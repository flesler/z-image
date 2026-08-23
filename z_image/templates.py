"""Deterministic `{a|b|c}` prompt templates resolved from seed."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

_GROUP_RE = re.compile(r"\{([^{}]+)\}")


@dataclass(frozen=True)
class ResolvedPrompt:
    template: str | None
    prompt: str


def _options_from_group(inner: str) -> list[str] | None:
    if "|" not in inner:
        return None
    options = [part.strip() for part in inner.split("|")]
    if len(options) < 2 or any(not option for option in options):
        return None
    return options


def has_placeholders(text: str) -> bool:
    for match in _GROUP_RE.finditer(text):
        if _options_from_group(match.group(1)):
            return True
    return False


def parse_groups(text: str) -> list[list[str]]:
    groups: list[list[str]] = []
    for match in _GROUP_RE.finditer(text):
        options = _options_from_group(match.group(1))
        if options:
            groups.append(options)
    return groups


def combo_count(groups: list[list[str]]) -> int:
    total = 1
    for group in groups:
        total *= len(group)
    return total


def resolve_prompt(text: str, seed: int, *, base_seed: int | None = None) -> ResolvedPrompt:
    groups = parse_groups(text)
    if not groups:
        return ResolvedPrompt(template=None, prompt=text)

    total = combo_count(groups)
    if base_seed is None:
        offset = seed % total
    else:
        offset = (seed - base_seed) % total
    indices: list[int] = []
    for group in groups:
        indices.append(offset % len(group))
        offset //= len(group)

    index = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal index
        options = _options_from_group(match.group(1))
        if not options:
            return match.group(0)
        choice = options[indices[index]]
        index += 1
        return choice

    resolved = _GROUP_RE.sub(repl, text)
    return ResolvedPrompt(template=text, prompt=resolved)


def all_variants(text: str, *, max_variants: int | None = None) -> list[str] | None:
    """All resolved prompts for a template (offset 0..n-1), or None if not a template."""
    groups = parse_groups(text)
    if not groups:
        return None
    limit = max_variants if max_variants is not None else int(
        os.environ.get("Z_IMAGE_TEMPLATE_MAX_VARIANTS", "64")
    )
    total = combo_count(groups)
    if total > limit:
        return None
    return [resolve_prompt(text, offset, base_seed=0).prompt for offset in range(total)]
