"""Tests for the Craftax LLM-first tutorial doc set.

Verifies:
1. All 10 chapter .md files exist.
2. Each chapter file is non-empty (> 100 chars).
3. compose_full() returns a string > 5000 chars.
4. compose(["overview"]) resolves to the overview anchor body.
5. compose(["nonexistent"]) raises ValueError.
"""
from __future__ import annotations

import pytest

from glyphbench.envs.craftax.docs import (
    ALL_SECTIONS,
    compose,
    compose_full,
    _CHAPTERS,
    _DOCS_DIR,
)


def test_all_chapter_files_exist() -> None:
    """All 10 chapter .md files must be present in the docs/ directory."""
    for chapter in _CHAPTERS:
        path = _DOCS_DIR / f"{chapter}.md"
        assert path.exists(), f"Missing chapter file: {path}"
        assert path.is_file(), f"Not a file: {path}"


def test_each_chapter_non_empty() -> None:
    """Each chapter must be more than 100 characters."""
    for chapter in _CHAPTERS:
        path = _DOCS_DIR / f"{chapter}.md"
        content = path.read_text(encoding="utf-8")
        assert len(content) > 100, (
            f"Chapter '{chapter}' is too short ({len(content)} chars)"
        )


def test_compose_full_length() -> None:
    """compose_full() must return a string longer than 5000 characters."""
    result = compose_full()
    assert isinstance(result, str)
    assert len(result) > 5000, (
        f"compose_full() returned only {len(result)} chars; expected > 5000"
    )


def test_compose_overview_anchor_resolves() -> None:
    """compose(['overview']) returns the overview anchor body, non-empty."""
    result = compose(["overview"])
    assert isinstance(result, str)
    assert "Craftax is a 9-floor survival crafting game" in result


def test_compose_unknown_anchor_raises() -> None:
    """compose(['nonexistent']) must raise ValueError."""
    with pytest.raises(ValueError, match="unknown"):
        compose(["nonexistent"])


def test_all_sections_resolve() -> None:
    """Every anchor in ALL_SECTIONS resolves without error."""
    body = compose(ALL_SECTIONS)
    assert body.strip() != ""
