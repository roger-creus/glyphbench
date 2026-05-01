"""GlyphbenchXMLParser: strict <action>NAME</action> only.

The parser extracts the LAST complete ``<action>NAME</action>`` match from
the model's response and verifies NAME against the env's ``ActionSpec``.
Any other shape (unclosed tag, JSON, bare-token, missing tag, unknown name)
forfeits the turn — see ``BaseGlyphEnv.forfeit_turn`` and the corresponding
behavior in ``verifiers_integration.env._apply_action_response``.

Last-match-wins is preserved so that models that quote ``<action>`` tags
inside their reasoning trace still get scored on the final committed tag.
"""

from __future__ import annotations

import re
from typing import Any

import verifiers as vf

from glyphbench.core.action import ActionSpec

NO_ACTION_TAG = "no_action_tag"
UNKNOWN_NAME = "unknown_name"

_XML_ACTION_RE = re.compile(
    r"<\s*action\s*>(.*?)<\s*/\s*action\s*>", re.DOTALL | re.IGNORECASE
)


class GlyphbenchXMLParser(vf.XMLParser):
    """Strict XML parser for the glyphbench verifiers integration.

    Returns ``(idx, name, parse_failed, parse_failure_reason)`` from
    ``parse_action``:

    - on success: ``(spec.index_of(name), spec.names[idx], False, None)``
    - on missing/unclosed/non-XML output:
      ``(noop_idx, noop_name, True, "no_action_tag")``
    - on tag found but NAME not in spec:
      ``(noop_idx, noop_name, True, "unknown_name")``

    The noop_idx/noop_name are returned for backward-compat with callers
    that need a non-None action; the verifiers integration ignores these
    and forfeits the turn instead.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            fields=["action"],
            answer_field="action",
            **kwargs,
        )

    def parse_action(
        self,
        raw_text: str,
        spec: ActionSpec,
        *,
        noop: str,
    ) -> tuple[int, str, bool, str | None]:
        text = raw_text or ""
        matches = _XML_ACTION_RE.findall(text)
        if not matches:
            return self._noop(spec, noop, NO_ACTION_TAG)
        candidate = matches[-1].strip()
        if not candidate:
            return self._noop(spec, noop, NO_ACTION_TAG)
        try:
            idx = spec.index_of(candidate)
        except KeyError:
            return self._noop(spec, noop, UNKNOWN_NAME)
        return idx, spec.names[idx], False, None

    @staticmethod
    def _noop(
        spec: ActionSpec, noop: str, reason: str
    ) -> tuple[int, str, bool, str]:
        try:
            idx = spec.index_of(noop)
        except KeyError:
            idx = 0
        return idx, spec.names[idx], True, reason
