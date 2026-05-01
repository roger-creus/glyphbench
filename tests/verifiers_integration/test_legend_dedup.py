"""System prompts must not duplicate the per-turn [Legend] block.

The per-turn legend (built inside each env's `_step`/`_render_*` and
extracted by `prompting.render_user_turn`) is the only legend shown to
the model. System prompts may describe the grid format conceptually,
but must not enumerate `<glyph> = <meaning>` lines.
"""

from __future__ import annotations

import pytest

from glyphbench.core.registry import all_glyphbench_env_ids, make_env
from glyphbench.envs import _import_all_suites


# Sentinel snippets that betray a legend enumeration in a system prompt.
# These are intentionally narrow: descriptive prose is fine, only enumerations
# matching `<char> [=:-] <description>` patterns trigger the failure.
_ENUMERATION_RE = (
    # e.g.  ☺ = you, ⇣ - stairs, P : player
    r"^\s*[^\w\s]\s*[=:\-]\s*\w",
)


@pytest.fixture(scope="module", autouse=True)
def _envs_loaded():
    _import_all_suites()


@pytest.mark.parametrize("env_id", sorted(
    eid for eid in all_glyphbench_env_ids() if "__dummy" not in eid
))
def test_system_prompt_does_not_enumerate_glyphs(env_id):
    import re
    env = make_env(env_id)
    sp = env.system_prompt()
    assert "[Legend]" not in sp, (
        f"{env_id}: system_prompt contains a [Legend] block — that header is "
        f"reserved for per-turn observations only."
    )
    # No `<symbol> = description` enumeration lines.
    pattern = re.compile(_ENUMERATION_RE[0], re.MULTILINE)
    matches = pattern.findall(sp)
    assert not matches, (
        f"{env_id}: system_prompt contains glyph-enumeration line(s): "
        f"{matches[:3]}. Move legend material to the per-turn observation."
    )
