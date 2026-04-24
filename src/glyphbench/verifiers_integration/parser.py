"""GlyphbenchXMLParser: XML-primary with JSON and bare-name fallbacks.

Wraps ``verifiers.XMLParser`` (fields ``think``, ``action``) and adds a
3-layer fallback chain that tolerates imperfectly-formatted model output:

    1. XML: ``<action>NAME</action>`` (preferred, last occurrence wins).
    2. JSON: ``{"action": "NAME"}`` inside any fence or top-level JSON object.
    3. Bare: last uppercase/snake-case token matching a known action name.

Unknown or missing action → ``(noop_idx, noop_name, parse_failed=True)``.
"""

from __future__ import annotations

import json
import re
from typing import Any

import verifiers as vf

from glyphbench.core.action import ActionSpec

_XML_ACTION_RE = re.compile(
    r"<\s*action\s*>(.*?)<\s*/\s*action\s*>", re.DOTALL | re.IGNORECASE
)
_XML_ACTION_OPEN_RE = re.compile(
    r"<\s*action\s*>(.*?)(?:<|\Z)", re.DOTALL | re.IGNORECASE
)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_BARE_NAME_RE = re.compile(r"\b([A-Z][A-Z0-9_]{1,})\b")


class GlyphbenchXMLParser(vf.XMLParser):
    """Parser used by the glyphbench verifiers integration.

    Instantiate with the default XML field layout (think + action); the extra
    ``parse_action`` method on top of the verifiers base provides the 3-layer
    fallback chain.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            fields=["think", "action"],
            answer_field="action",
            **kwargs,
        )

    def parse_action(
        self,
        raw_text: str,
        spec: ActionSpec,
        *,
        noop: str,
    ) -> tuple[int, str, bool]:
        """Extract an action from model output.

        Returns ``(action_idx, canonical_action_name, parse_failed)``.

        The canonical action name is always one of ``spec.names`` — on any
        parse failure or unknown action name we fall back to ``noop`` and
        flag ``parse_failed=True``.
        """
        candidate = self._extract_candidate(raw_text or "")
        if candidate is None:
            return self._noop(spec, noop)
        try:
            return (spec.index_of(candidate), spec.names[spec.index_of(candidate)], False)
        except KeyError:
            return self._noop(spec, noop)

    def _extract_candidate(self, text: str) -> str | None:
        # 1. Complete <action>...</action> — take the last occurrence.
        matches = _XML_ACTION_RE.findall(text)
        if matches:
            return matches[-1].strip()

        # 2. Unclosed <action>... — tolerant.
        open_match = _XML_ACTION_OPEN_RE.search(text)
        if open_match:
            cand = open_match.group(1).strip()
            # Only use if it looks like a plain action name (no nested XML).
            if cand and "<" not in cand and len(cand) < 64:
                return cand

        # 3. JSON — fenced block, then top-level object scan.
        for body in _iter_json_bodies(text):
            try:
                obj = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict) and isinstance(obj.get("action"), str):
                return obj["action"].strip()

        # 4. Bare uppercase-token fallback — take the last one.
        bare = _BARE_NAME_RE.findall(text)
        if bare:
            return bare[-1].strip()

        return None

    @staticmethod
    def _noop(spec: ActionSpec, noop: str) -> tuple[int, str, bool]:
        try:
            idx = spec.index_of(noop)
        except KeyError:
            # Very defensive — an env without its declared noop: fall back to idx 0.
            idx = 0
        return idx, spec.names[idx], True


def _iter_json_bodies(text: str):
    """Yield candidate JSON bodies from fenced code blocks + top-level scan."""
    for m in _FENCE_RE.finditer(text):
        yield m.group(1)
    # Brace-balanced scan.
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
                        yield text[start : j + 1]
                        break
            j += 1
        i = j + 1
