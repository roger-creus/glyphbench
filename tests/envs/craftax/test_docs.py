"""Tests for the Craftax LLM-first tutorial doc set (T_FINAL).

Verifies:
1. All 10 chapter .md files exist.
2. Each chapter file is non-empty (> 100 chars).
3. compose_full() returns a string > 5000 chars.
4. compose(["overview"]) returns exactly the overview chapter.
5. compose(["nonexistent"]) raises ValueError.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from glyphbench.envs.craftax.docs import (
    ALL_CHAPTERS,
    compose,
    compose_full,
    _DOCS_DIR,
)


def test_all_chapter_files_exist() -> None:
    """All 10 chapter .md files must be present in the docs/ directory."""
    for chapter in ALL_CHAPTERS:
        path = _DOCS_DIR / f"{chapter}.md"
        assert path.exists(), f"Missing chapter file: {path}"
        assert path.is_file(), f"Not a file: {path}"


def test_each_chapter_non_empty() -> None:
    """Each chapter must be more than 100 characters."""
    for chapter in ALL_CHAPTERS:
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


def test_compose_single_chapter() -> None:
    """compose(["overview"]) must return exactly the overview chapter content."""
    overview_path = _DOCS_DIR / "overview.md"
    expected = overview_path.read_text(encoding="utf-8")
    result = compose(["overview"])
    assert result == expected, (
        "compose(['overview']) did not return the raw overview.md content"
    )


def test_compose_unknown_chapter_raises() -> None:
    """compose(["nonexistent"]) must raise ValueError."""
    with pytest.raises(ValueError, match="unknown chapter"):
        compose(["nonexistent"])
