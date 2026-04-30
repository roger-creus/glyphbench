"""Craftax LLM-first tutorial doc set + per-task composer.

Each .md file in this package is a self-contained chapter. The composer
joins selected chapters into a single system-prompt string for a given
task. The full env (`craftax-v0`) uses ALL chapters; per-subtask envs
include only the relevant ones.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

_DOCS_DIR = Path(__file__).parent

# Canonical chapter ordering — used by the full env.
ALL_CHAPTERS: tuple[str, ...] = (
    "overview",
    "legend",
    "survival",
    "combat",
    "crafting",
    "magic",
    "items",
    "progression",
    "floors",
    "boss",
)


def _read_chapter(name: str) -> str:
    path = _DOCS_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


def compose(chapters: Iterable[str] = ALL_CHAPTERS) -> str:
    """Compose a system prompt by concatenating selected chapters in order."""
    parts: list[str] = []
    for ch in chapters:
        if ch not in ALL_CHAPTERS:
            raise ValueError(f"unknown chapter {ch!r}; valid: {ALL_CHAPTERS}")
        parts.append(_read_chapter(ch))
    return "\n\n".join(parts)


def compose_full() -> str:
    """Convenience: return the full tutorial joined."""
    return compose(ALL_CHAPTERS)
