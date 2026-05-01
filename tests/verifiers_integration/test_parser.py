"""Tests for GlyphbenchXMLParser: strict <action>NAME</action> only."""

from __future__ import annotations

import pytest

from glyphbench.core.action import ActionSpec
from glyphbench.verifiers_integration.parser import (
    GlyphbenchXMLParser,
    NO_ACTION_TAG,
    UNKNOWN_NAME,
)


@pytest.fixture
def spec():
    return ActionSpec(
        names=("LEFT", "RIGHT", "UP", "DOWN", "NOOP"),
        descriptions=("l", "r", "u", "d", "n"),
    )


@pytest.fixture
def parser():
    return GlyphbenchXMLParser()


def test_well_formed_xml(parser, spec):
    text = "<think>reason</think><action>LEFT</action>"
    idx, name, failed, reason = parser.parse_action(text, spec, noop="NOOP")
    assert (idx, name, failed, reason) == (0, "LEFT", False, None)


def test_xml_case_insensitive(parser, spec):
    text = "<think>x</think><action>right</action>"
    idx, name, failed, reason = parser.parse_action(text, spec, noop="NOOP")
    assert (idx, name, failed, reason) == (1, "RIGHT", False, None)


def test_xml_whitespace_in_action(parser, spec):
    text = "<think>x</think><action>   UP\n</action>"
    idx, name, failed, reason = parser.parse_action(text, spec, noop="NOOP")
    assert (idx, name, failed, reason) == (2, "UP", False, None)


def test_multiple_action_tags_take_last(parser, spec):
    # Models often quote <action> tags during reasoning; only the
    # final committed tag counts.
    text = "<action>LEFT</action>some reasoning<action>RIGHT</action>"
    idx, name, failed, reason = parser.parse_action(text, spec, noop="NOOP")
    assert (name, failed, reason) == ("RIGHT", False, None)


def test_unknown_action_name_forfeits(parser, spec):
    text = "<think>x</think><action>FLY</action>"
    idx, name, failed, reason = parser.parse_action(text, spec, noop="NOOP")
    assert idx == spec.index_of("NOOP")
    assert name == "NOOP"  # noop name returned for legacy step compat
    assert failed is True
    assert reason == UNKNOWN_NAME


def test_missing_action_tag_forfeits(parser, spec):
    text = "<think>i have no action</think>"
    idx, name, failed, reason = parser.parse_action(text, spec, noop="NOOP")
    assert failed is True
    assert reason == NO_ACTION_TAG


def test_empty_string_forfeits(parser, spec):
    idx, name, failed, reason = parser.parse_action("", spec, noop="NOOP")
    assert failed is True
    assert reason == NO_ACTION_TAG


def test_unclosed_action_tag_forfeits(parser, spec):
    # Layer-2 tolerance removed: <action>UP without </action> no longer parses.
    text = "<action>UP"
    idx, name, failed, reason = parser.parse_action(text, spec, noop="NOOP")
    assert failed is True
    assert reason == NO_ACTION_TAG


def test_json_output_forfeits(parser, spec):
    # Layer-3 fallback removed.
    text = '{"action": "DOWN"}'
    idx, name, failed, reason = parser.parse_action(text, spec, noop="NOOP")
    assert failed is True
    assert reason == NO_ACTION_TAG


def test_fenced_json_forfeits(parser, spec):
    text = "```json\n{\"action\": \"LEFT\"}\n```"
    idx, name, failed, reason = parser.parse_action(text, spec, noop="NOOP")
    assert failed is True
    assert reason == NO_ACTION_TAG


def test_bare_token_forfeits(parser, spec):
    # Layer-4 fallback removed.
    text = "I will go UP"
    idx, name, failed, reason = parser.parse_action(text, spec, noop="NOOP")
    assert failed is True
    assert reason == NO_ACTION_TAG


def test_action_quoted_inside_think_then_committed(parser, spec):
    # Common pattern: model quotes a candidate inside <think> then commits
    # outside. Only the last <action> tag wins, so the committed one applies.
    text = "<think>Maybe <action>LEFT</action>?</think><action>UP</action>"
    idx, name, failed, reason = parser.parse_action(text, spec, noop="NOOP")
    assert (name, failed, reason) == ("UP", False, None)
