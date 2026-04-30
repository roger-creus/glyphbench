"""Per-env tutorial-coverage gates.

Every craftax env must:
  1. Resolve every entry in `tutorial_sections` via compose() without error.
  2. Produce a non-empty system_prompt() that contains the env id, the
     "TASK" marker, and the actions header.
  3. Cover every action in self.action_spec via tutorial_sections (per
     ACTION_TO_ANCHORS).
  4. Cover every glyph the env's grid can render via tutorial_sections (per
     GLYPH_TO_ANCHORS).
"""
from __future__ import annotations

import pytest

# Importing the suite registers all envs.
import glyphbench.envs.craftax  # noqa: F401
from glyphbench.core.registry import REGISTRY
from glyphbench.envs.craftax.docs import ALL_SECTIONS, _resolve, compose
from glyphbench.envs.craftax.docs._coverage import (
    ACTION_TO_ANCHORS,
    GLYPH_TO_ANCHORS,
)


CRAFTAX_ENV_IDS = sorted(eid for eid in REGISTRY if eid.startswith("glyphbench/craftax-"))


def _expand_sections(sections: tuple[str, ...]) -> set[str]:
    """Expand bare chapter names + dedup to a set of leaf anchor names."""
    out: set[str] = set()
    for s in sections:
        for name in _resolve(s):
            out.add(name)
    return out


@pytest.mark.parametrize("env_id", CRAFTAX_ENV_IDS)
def test_tutorial_sections_resolve(env_id: str):
    cls = REGISTRY[env_id]
    sections = getattr(cls, "tutorial_sections", None)
    assert sections is not None, f"{env_id} missing tutorial_sections"
    assert isinstance(sections, tuple) and len(sections) > 0, (
        f"{env_id}.tutorial_sections must be a non-empty tuple"
    )
    body = compose(sections)
    assert body.strip() != ""


@pytest.mark.parametrize("env_id", CRAFTAX_ENV_IDS)
def test_system_prompt_smoke(env_id: str):
    env = REGISTRY[env_id]()
    env.reset(seed=0)
    prompt = env.system_prompt()
    # Every prompt must mention the env id (or its short form).
    assert env_id in prompt or env.env_id() in prompt
    assert "TASK" in prompt
    # Action-list header from ActionSpec.render_for_prompt().
    assert "Actions" in prompt or "action" in prompt.lower()
    # Every action name must appear verbatim in the prompt body.
    for name in env.action_spec.names:
        assert name in prompt, (
            f"{env_id}: action {name!r} missing from system_prompt"
        )


@pytest.mark.parametrize("env_id", CRAFTAX_ENV_IDS)
def test_action_coverage(env_id: str):
    cls = REGISTRY[env_id]
    sections = _expand_sections(cls.tutorial_sections)
    env = cls()
    for action in env.action_spec.names:
        anchors = ACTION_TO_ANCHORS.get(action)
        assert anchors is not None, (
            f"action {action!r} unknown to ACTION_TO_ANCHORS registry"
        )
        assert anchors & sections, (
            f"{env_id}: action {action!r} not covered. "
            f"Need any of {sorted(anchors)}, env has {sorted(sections)}"
        )


@pytest.mark.parametrize("env_id", CRAFTAX_ENV_IDS)
def test_glyph_coverage(env_id: str):
    cls = REGISTRY[env_id]
    sections = _expand_sections(cls.tutorial_sections)
    env = cls()
    env.reset(seed=0)
    obs = env.get_observation()
    glyphs = _glyphs_in_grid(obs.grid)
    for glyph in glyphs:
        anchors = GLYPH_TO_ANCHORS.get(glyph)
        if anchors is None:
            pytest.fail(
                f"{env_id}: glyph {glyph!r} (U+{ord(glyph):04X}) "
                f"unknown to GLYPH_TO_ANCHORS registry"
            )
        assert anchors & sections, (
            f"{env_id}: glyph {glyph!r} (U+{ord(glyph):04X}) not covered. "
            f"Need any of {sorted(anchors)}, env has {sorted(sections)}"
        )


def _glyphs_in_grid(grid: str) -> set[str]:
    """Return the set of single-codepoint non-whitespace glyphs in `grid`."""
    return {ch for ch in grid if not ch.isspace()}


def test_action_registry_covers_full_spec():
    """ACTION_TO_ANCHORS must know every action in CRAFTAX_FULL_ACTION_SPEC and
    CRAFTAX_ACTION_SPEC."""
    from glyphbench.envs.craftax.base import (
        CRAFTAX_ACTION_SPEC,
        CRAFTAX_FULL_ACTION_SPEC,
    )
    all_actions = set(CRAFTAX_ACTION_SPEC.names) | set(CRAFTAX_FULL_ACTION_SPEC.names)
    for action in all_actions:
        assert action in ACTION_TO_ANCHORS, (
            f"action {action!r} not in ACTION_TO_ANCHORS registry"
        )


def test_glyph_registry_targets_real_anchors():
    """Every anchor cited by GLYPH_TO_ANCHORS must be a real ALL_SECTIONS entry."""
    valid = set(ALL_SECTIONS)
    for glyph, anchors in GLYPH_TO_ANCHORS.items():
        for anchor in anchors:
            assert anchor in valid, (
                f"glyph {glyph!r} cites non-existent anchor {anchor!r}"
            )


def test_action_registry_targets_real_anchors():
    """Every anchor cited by ACTION_TO_ANCHORS must be a real ALL_SECTIONS entry."""
    valid = set(ALL_SECTIONS)
    for action, anchors in ACTION_TO_ANCHORS.items():
        for anchor in anchors:
            assert anchor in valid, (
                f"action {action!r} cites non-existent anchor {anchor!r}"
            )
