"""System prompts must not duplicate the per-turn [Legend] block.

The per-turn legend (built inside each env's `_step`/`_render_*` and
extracted by `prompting.render_user_turn`) is the only legend shown to
the model. System prompts may describe the grid format conceptually,
but must not enumerate `<glyph> = <meaning>` lines.
"""

from __future__ import annotations

import re

import pytest

from glyphbench.core.registry import all_glyphbench_env_ids, make_env
from glyphbench.envs import _import_all_suites


# A legend is at least 3 consecutive lines of `<glyph><sep><description>`
# where <glyph> is a single non-word character and <sep> is = or : or –.
# Isolated bullets like "- -1 reward" no longer false-positive.
_LEGEND_LINE_RE = re.compile(r"^\s+\S\s+[=:\-]\s+\S", re.MULTILINE)


def _has_legend_block(text: str) -> bool:
    lines = text.splitlines()
    streak = 0
    for line in lines:
        if _LEGEND_LINE_RE.match(line):
            streak += 1
            if streak >= 3:
                return True
        else:
            streak = 0
    return False


@pytest.fixture(scope="module", autouse=True)
def _envs_loaded():
    _import_all_suites()


@pytest.mark.parametrize("env_id", sorted(
    eid for eid in all_glyphbench_env_ids() if "__dummy" not in eid
))
def test_system_prompt_does_not_enumerate_glyphs(env_id):
    env = make_env(env_id)
    sp = env.system_prompt()
    assert "[Legend]" not in sp, (
        f"{env_id}: system_prompt contains a [Legend] block — that header is "
        f"reserved for per-turn observations only."
    )
    # No legend-block enumerations (3+ consecutive `<glyph> <sep> <desc>` lines).
    assert not _has_legend_block(sp), (
        f"{env_id}: system_prompt contains a glyph-enumeration legend block. "
        f"Move legend material to the per-turn observation."
    )
