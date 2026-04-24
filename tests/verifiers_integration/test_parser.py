"""Tests for GlyphbenchXMLParser: XML-primary with JSON/regex fallback."""

from __future__ import annotations

import pytest

from glyphbench.core.action import ActionSpec
from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser


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
    idx, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert (idx, name, failed) == (0, "LEFT", False)


def test_xml_case_insensitive(parser, spec):
    text = "<think>x</think><action>right</action>"
    idx, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert (idx, name, failed) == (1, "RIGHT", False)


def test_xml_whitespace_in_action(parser, spec):
    text = "<think>x</think><action>   UP\n</action>"
    idx, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert (idx, name, failed) == (2, "UP", False)


def test_multiple_action_tags_take_last(parser, spec):
    text = "<action>LEFT</action>some text<action>RIGHT</action>"
    idx, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert name == "RIGHT"


def test_unknown_action_falls_back_to_noop(parser, spec):
    text = "<think>x</think><action>FLY</action>"
    idx, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert (name, failed) == ("NOOP", True)


def test_missing_action_tag_falls_back(parser, spec):
    text = "<think>i have no action</think>"
    _, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert (name, failed) == ("NOOP", True)


def test_empty_string_falls_back(parser, spec):
    _, name, failed = parser.parse_action("", spec, noop="NOOP")
    assert (name, failed) == ("NOOP", True)


def test_json_fallback(parser, spec):
    text = 'no xml, but: {"thinking":"x","action":"DOWN"}'
    idx, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert (idx, name, failed) == (3, "DOWN", False)


def test_json_fenced_fallback(parser, spec):
    text = '```json\n{"action": "LEFT"}\n```'
    _, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert (name, failed) == ("LEFT", False)


def test_bare_action_name_fallback(parser, spec):
    # Last-resort: the response contains nothing but an action name.
    text = "UP"
    _, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert (name, failed) == ("UP", False)


def test_prose_mentioning_action_does_not_trigger_bare_fallback(parser, spec):
    # "I will go LEFT" should still parse via no-xml fallback chain — if the
    # bare-name regex matches the only action-looking token, that's fine.
    # This test documents the choice: the last-ditch regex picks the last
    # uppercase token matching an action name.
    text = "I am considering going LEFT or RIGHT. Final answer: DOWN"
    _, name, _ = parser.parse_action(text, spec, noop="NOOP")
    assert name == "DOWN"


def test_malformed_xml_nothing_else_works(parser, spec):
    text = "<action>LEFT"  # no close tag, no json, no bare action at end
    _, name, failed = parser.parse_action(text, spec, noop="NOOP")
    # Our fallback chain should still find "LEFT" via the bare-name regex.
    assert name == "LEFT"
    assert failed is False


def test_completely_off_the_rails(parser, spec):
    text = "asdfjkl; qwerty 12345"
    _, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert (name, failed) == ("NOOP", True)


def test_get_format_reward_func_is_callable(parser):
    # verifiers XMLParser provides a format-reward fn; we expose it.
    fn = parser.get_format_reward_func()
    assert callable(fn)
