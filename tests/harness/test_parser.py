import pytest

from glyphbench.core.action import ActionSpec
from glyphbench.harness.parser import (
    MAX_REPAIR_RETRIES,
    extract_json,
    parse_harness_output,
)

ACTION_SPEC = ActionSpec(
    names=("NORTH", "SOUTH", "EAST", "WEST", "NOOP"),
    descriptions=("", "", "", "", ""),
)


def test_extract_json_bare():
    text = '{"action": "NORTH"}'
    assert extract_json(text) == '{"action": "NORTH"}'


def test_extract_json_wrapped_in_fence():
    text = 'some prose\n```json\n{"action": "NORTH"}\n```\nmore prose'
    assert extract_json(text).strip() == '{"action": "NORTH"}'


def test_extract_json_wrapped_in_plain_fence():
    text = '```\n{"action": "NORTH"}\n```'
    assert extract_json(text).strip() == '{"action": "NORTH"}'


def test_extract_json_no_json_raises():
    with pytest.raises(ValueError, match="no JSON"):
        extract_json("I refuse to emit JSON.")


def test_extract_json_picks_last_object_when_multiple():
    text = '{"a": 1} then {"action": "NORTH"}'
    result = extract_json(text)
    # Acceptable: either the first or the one containing "action" — we want the LAST top-level object
    assert '"action": "NORTH"' in result


def test_parse_harness_output_valid():
    text = '{"action": "NORTH", "thinking": "go up"}'
    result = parse_harness_output(text, ACTION_SPEC, noop_action_name="NOOP")
    assert result.action_index == 0
    assert result.action_name == "NORTH"
    assert result.parse_error is None
    assert result.fell_back_to_noop is False


def test_parse_harness_output_invalid_json_no_repair_available_returns_error():
    text = "not json at all"
    result = parse_harness_output(text, ACTION_SPEC, noop_action_name="NOOP")
    assert result.parse_error is not None
    assert result.fell_back_to_noop is True
    assert result.action_name == "NOOP"


def test_parse_harness_output_invalid_action_name_returns_error():
    text = '{"action": "FLY"}'
    result = parse_harness_output(text, ACTION_SPEC, noop_action_name="NOOP")
    assert result.parse_error is not None
    assert "FLY" in result.parse_error
    assert result.fell_back_to_noop is True


def test_parse_harness_output_missing_action_field():
    text = '{"thinking": "no action"}'
    result = parse_harness_output(text, ACTION_SPEC, noop_action_name="NOOP")
    assert result.parse_error is not None
    assert result.fell_back_to_noop is True


def test_max_repair_retries_is_three():
    assert MAX_REPAIR_RETRIES == 3


def test_parser_falls_back_on_lowercase():
    text = '{"action": "north", "thinking": "go up"}'
    result = parse_harness_output(text, ACTION_SPEC, noop_action_name="NOOP")
    assert result.parse_error is None
    assert result.fell_back_to_noop is False
    assert result.action_index == 0
    assert result.action_name == "NORTH"
