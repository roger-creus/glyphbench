"""Tests that the MiniGrid system prompt stays in sync with rendered glyphs.

Bug being regression-tested: the base-class system prompt described symbols
like ``#``, ``>v<^``, ``G``, ``.`` as walls/agent/goal/floor, but the actual
renderer emits Unicode glyphs (``█``, ``→↓←↑``, ``★``, ``·``). LLMs read the
prompt, then see different characters in observations, and get confused.

These tests assert the bidirectional invariant:
  1. Every non-trivial glyph that appears in a rendered observation also
     appears somewhere in ``env.system_prompt()``.
  2. Every single-character non-trivial glyph mentioned as a symbol in the
     system prompt actually shows up in a rendered observation (no orphan
     references to symbols the renderer never emits).

Runs across 3+ MiniGrid variants: Empty-5x5, DoorKey-5x5, Crossing-N1 (lava).
"""

from __future__ import annotations

import pytest

from glyphbench.envs.minigrid.base import DIR_TO_CHAR
from glyphbench.envs.minigrid.crossing import MiniGridCrossingN1Env
from glyphbench.envs.minigrid.doorkey import MiniGridDoorKey5x5Env
from glyphbench.envs.minigrid.empty import MiniGridEmpty5x5Env
from glyphbench.envs.minigrid.objects import Goal, Lava, Wall, Water

# Trivial characters we don't want to gate on (JSON/format punctuation,
# whitespace, common ASCII letters used in English words in the prompt).
_TRIVIAL = set(" \t\n\r{}[](),.:;!?/\\'\"`-_=+*%@#&^<>|~$")


def _rendered_glyphs(env) -> set[str]:
    """Collect every non-trivial, non-alphanumeric char from a rendering.

    Uses multiple seeds so we see every branch (e.g. DoorKey with key in
    different positions, Crossing gap placements)."""
    seen: set[str] = set()
    for seed in range(5):
        env.reset(seed=seed)
        obs = env.get_observation()
        rendered = obs.render()
        for ch in rendered:
            if ch.isalnum() or ch in _TRIVIAL:
                continue
            seen.add(ch)
    return seen


def _prompt_chars(env) -> str:
    """The system prompt text, as one string."""
    return env.system_prompt()


@pytest.mark.parametrize(
    "make_env",
    [
        lambda: MiniGridEmpty5x5Env(max_turns=20),
        lambda: MiniGridDoorKey5x5Env(max_turns=20),
        lambda: MiniGridCrossingN1Env(max_turns=20),
    ],
    ids=["empty-5x5", "doorkey-5x5", "crossing-n1-lava"],
)
class TestMiniGridPromptSync:
    def test_every_rendered_glyph_appears_in_prompt(self, make_env) -> None:
        """Any Unicode glyph the renderer emits must be documented in the
        system prompt (in the text body or legend)."""
        env = make_env()
        rendered = _rendered_glyphs(env)
        prompt = _prompt_chars(env)
        missing = {g for g in rendered if g not in prompt}
        assert not missing, (
            f"Glyphs rendered in observations but NOT present in system prompt: "
            f"{sorted(missing)}. The prompt must document every glyph the agent "
            f"will see."
        )

    def test_no_stale_ascii_placeholders(self, make_env) -> None:
        """The prompt must not describe old ASCII placeholders that the
        renderer no longer emits. These exact symbols (``#``, ``>``, ``v``,
        ``<``, ``^``, ``G`` as a glyph label, ``.``, ``L``, ``K``, ``~``) were
        the pre-fix hardcoded placeholders — the renderer uses
        █/→↓←↑/★/·/♧/♠/≈ instead. Forbid them as quoted single-character
        glyph labels in the prompt."""
        env = make_env()
        prompt = _prompt_chars(env)
        # Forbid legacy parenthesized single-char glyph labels like " (G)" or
        # " (L)" or " (#)". These are the exact patterns from the bug report.
        bad_labels = [
            "(G)", "(L)", "(K)", "(#)", "(~)", "(.)", "(>)", "(<)", "(^)", "(v)",
        ]
        for bad in bad_labels:
            assert bad not in prompt, (
                f"System prompt contains legacy ASCII glyph label {bad!r}. "
                f"The renderer does not emit this character; reference the "
                f"actual Unicode glyph instead."
            )

    def test_agent_arrow_glyphs_all_present(self, make_env) -> None:
        """All four agent-direction glyphs must be in the prompt."""
        env = make_env()
        prompt = _prompt_chars(env)
        for d, ch in DIR_TO_CHAR.items():
            assert ch in prompt, f"Agent direction glyph {ch!r} (dir={d}) missing"

    def test_wall_and_goal_glyphs_in_prompt(self, make_env) -> None:
        """Wall and Goal glyphs always present (every env has walls + goal)."""
        env = make_env()
        prompt = _prompt_chars(env)
        assert Wall().render_char() in prompt, "Wall glyph (█) missing"
        assert Goal().render_char() in prompt, "Goal glyph (★) missing"


class TestPromptReferencesRealGlyphs:
    """Env-specific: make sure lava / water are documented when present."""

    def test_lava_glyph_in_crossing_prompt(self) -> None:
        env = MiniGridCrossingN1Env(max_turns=20)
        prompt = env.system_prompt()
        assert Lava().render_char() in prompt, (
            f"Lava glyph {Lava().render_char()!r} missing from Crossing prompt"
        )

    def test_water_glyph_in_safe_crossing_prompt(self) -> None:
        from glyphbench.envs.minigrid.crossing import MiniGridCrossingN1SafeEnv

        env = MiniGridCrossingN1SafeEnv(max_turns=20)
        prompt = env.system_prompt()
        assert Water().render_char() in prompt, (
            f"Water glyph {Water().render_char()!r} missing from safe-crossing prompt"
        )


class TestAllMinigridEnvsConsistent:
    """Smoke-check every registered minigrid env (all 71) to guarantee that
    the universal glyphs (walls, floor, agent arrows, goal) are documented
    in each system prompt."""

    def _all_env_ids(self) -> list[str]:
        import gymnasium as gym

        import glyphbench  # noqa: F401  # triggers env registration

        return sorted(
            eid for eid in gym.registry.keys()
            if eid.startswith("glyphbench/minigrid-")
        )

    def test_seventy_one_envs_registered(self) -> None:
        assert len(self._all_env_ids()) == 71

    def test_universal_glyphs_in_every_prompt(self) -> None:
        """Wall (█), all four agent arrows, and the goal (★) must appear in
        every minigrid env's system prompt."""
        import gymnasium as gym

        wall = Wall().render_char()
        goal = Goal().render_char()
        arrows = tuple(DIR_TO_CHAR.values())
        missing: dict[str, list[str]] = {}
        for env_id in self._all_env_ids():
            env = gym.make(env_id, max_turns=20).unwrapped
            prompt = env.system_prompt()
            miss = []
            if wall not in prompt:
                miss.append(f"wall {wall!r}")
            if goal not in prompt:
                miss.append(f"goal {goal!r}")
            for a in arrows:
                if a not in prompt:
                    miss.append(f"arrow {a!r}")
            if miss:
                missing[env_id] = miss
        assert not missing, (
            f"Envs with missing universal glyphs in system_prompt(): {missing}"
        )
