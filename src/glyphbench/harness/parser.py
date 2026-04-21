"""JSON extraction, validation, and parse result for the harness.

This module is synchronous and has no LLM dependency. The actual retry loop
that calls a repair prompt lives in `agent.py`, which calls `parse_harness_output`
repeatedly on each attempt.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from glyphbench.core.action import ActionSpec
from glyphbench.harness.schema import HarnessOutput

MAX_REPAIR_RETRIES = 3

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _find_json_objects(text: str) -> list[str]:
    """Return all top-level {...} substrings in `text` that parse as JSON.

    Uses a brace-balanced scan rather than a regex so we correctly handle
    multiple separate objects and nested braces. String-literal-aware to
    avoid mistaking `{` or `}` inside a JSON string for structural braces.
    """
    results: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escape = False
        start = i
        j = i
        while j < n:
            ch = text[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : j + 1]
                        try:
                            json.loads(candidate)
                            results.append(candidate)
                        except json.JSONDecodeError:
                            pass
                        break
            j += 1
        i = j + 1
    return results


def extract_json(text: str) -> str:
    """Pull the JSON object out of the LLM's raw response.

    Tries in order:
      1. Content inside a ```json ... ``` or ``` ... ``` fence.
      2. The last top-level {...} block in the text that parses as JSON.
      3. The entire stripped text, if it parses as JSON.

    Raises ValueError if no JSON object is found.
    """
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass  # fall through to object-scan fallback

    # Find all top-level {...} blocks and take the last one that parses.
    objects = _find_json_objects(text)
    if objects:
        return objects[-1]

    stripped = text.strip()
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    raise ValueError("no JSON object found in LLM response")


@dataclass
class ParseResult:
    """The result of parsing one LLM response attempt.

    If parsing succeeded, `parsed` is populated and `parse_error` is None.
    If parsing failed and we exhausted retries, `parse_error` describes the
    failure, `fell_back_to_noop` is True, and `action_name` is the noop action.
    """

    action_index: int
    action_name: str
    parsed: HarnessOutput | None
    retries_used: int
    fell_back_to_noop: bool
    parse_error: str | None
    raw_responses: list[str] = field(default_factory=list)


def parse_harness_output(
    raw_text: str,
    action_spec: ActionSpec,
    *,
    noop_action_name: str,
) -> ParseResult:
    """Parse a single LLM response.

    This function does not retry; the retry loop lives in `agent.py`. This
    function is synchronous, pure, and easy to unit test.

    On any failure — invalid JSON, schema mismatch, unknown action name — the
    result has `parse_error` populated and falls back to the noop action.
    """
    try:
        json_text = extract_json(raw_text)
    except ValueError as e:
        return _noop_result(action_spec, noop_action_name, f"extract_json failed: {e}")

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        return _noop_result(action_spec, noop_action_name, f"json.loads failed: {e}")

    try:
        parsed = HarnessOutput.model_validate(data)
    except ValidationError as e:
        return _noop_result(action_spec, noop_action_name, f"schema validation failed: {e}")

    action_name = parsed.action
    try:
        action_index = action_spec.index_of(action_name)
    except KeyError as e:
        return _noop_result(action_spec, noop_action_name, f"unknown action: {action_name!r} ({e})")

    canonical_name = action_spec.names[action_index]
    return ParseResult(
        action_index=action_index,
        action_name=canonical_name,
        parsed=parsed,
        retries_used=0,
        fell_back_to_noop=False,
        parse_error=None,
        raw_responses=[raw_text],
    )


def _noop_result(action_spec: ActionSpec, noop_action_name: str, error: str) -> ParseResult:
    noop_index = action_spec.index_of(noop_action_name)
    return ParseResult(
        action_index=noop_index,
        action_name=noop_action_name,
        parsed=None,
        retries_used=0,
        fell_back_to_noop=True,
        parse_error=error,
    )
