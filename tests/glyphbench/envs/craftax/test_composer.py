"""Anchor-aware composer unit tests."""
from __future__ import annotations

import pytest

from glyphbench.envs.craftax.docs import (
    ALL_SECTIONS,
    compose,
    parse_anchors,
)


def test_compose_unknown_anchor_raises():
    with pytest.raises(ValueError, match="unknown"):
        compose(["does:not_exist"])


def test_compose_unknown_chapter_raises():
    with pytest.raises(ValueError, match="unknown"):
        compose(["nope"])


def test_compose_unknown_anchor_in_known_chapter_raises():
    with pytest.raises(ValueError, match="unknown anchor"):
        compose(["combat:does_not_exist"])


def test_compose_returns_non_empty_string():
    result = compose(["overview"])
    assert isinstance(result, str)
    assert result.strip() != ""


def test_compose_idempotent():
    a = compose(["overview", "boss"])
    b = compose(["overview", "boss"])
    assert a == b


def test_compose_preserves_order():
    forward = compose(["overview", "boss"])
    reverse = compose(["boss", "overview"])
    assert forward != reverse


def test_compose_all_sections_resolves():
    result = compose(ALL_SECTIONS)
    assert result.strip() != ""
    # Sanity: should contain content from every chapter.
    assert "Overworld" in result
    assert "Necromancer" in result
    assert "Mana" in result


def test_compose_chapter_name_expands_to_all_anchors_in_file():
    legend_full = compose(["legend"])
    legend_player = compose(["legend:player"])
    assert legend_player in legend_full
    legend_hud = compose(["legend:hud"])
    assert legend_hud in legend_full


def test_compose_dedupes_repeat_specs():
    once = compose(["overview"])
    twice = compose(["overview", "overview"])
    assert once == twice


def test_parse_anchors_strips_surrounding_blank_lines():
    src = (
        "<!-- :section foo -->\n"
        "\n"
        "body\n"
        "\n"
        "<!-- :end -->\n"
    )
    anchors = parse_anchors(src)
    assert anchors == {"foo": "body"}


def test_parse_anchors_rejects_duplicate():
    src = (
        "<!-- :section foo -->\nbody1\n<!-- :end -->\n"
        "<!-- :section foo -->\nbody2\n<!-- :end -->\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        parse_anchors(src)


def test_parse_anchors_rejects_unclosed():
    src = "<!-- :section foo -->\nbody\n"
    with pytest.raises(ValueError, match="unclosed"):
        parse_anchors(src)


def test_parse_anchors_handles_multiple():
    src = (
        "<!-- :section foo -->\nbody1\n<!-- :end -->\n"
        "intermission\n"
        "<!-- :section bar -->\nbody2\n<!-- :end -->\n"
    )
    anchors = parse_anchors(src)
    assert anchors == {"foo": "body1", "bar": "body2"}


def test_all_sections_are_valid():
    """ALL_SECTIONS only contains anchors that resolve in their chapter files."""
    body = compose(ALL_SECTIONS)
    assert body.strip() != ""


def test_chapter_files_have_no_duplicate_anchors():
    """No chapter file may declare the same anchor name twice."""
    from glyphbench.envs.craftax.docs import _CHAPTERS, _chapter_anchors

    for chapter in _CHAPTERS:
        anchors = _chapter_anchors(chapter)  # raises on duplicate
        assert isinstance(anchors, dict)
        assert len(anchors) > 0, f"chapter {chapter!r} has no anchors"
