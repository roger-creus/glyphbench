"""Tests for prompting.py: system prompt + frame-stack user-turn rendering."""

from __future__ import annotations

from collections import deque

import pytest

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation
from glyphbench.verifiers_integration.prompting import (
    build_system_prompt,
    render_user_turn,
)


class _Game(BaseGlyphEnv):
    action_spec = ActionSpec(
        names=("LEFT", "RIGHT", "UP", "DOWN"),
        descriptions=("l", "r", "u", "d"),
    )

    def _reset(self, seed: int) -> GridObservation:
        return GridObservation(grid="A.", legend="A symbol-a", hud="step=0", message="")

    def _step(self, action: int):
        return (
            GridObservation(grid="A.", legend="A symbol-a", hud="step=1", message=""),
            0.0,
            False,
            False,
            {},
        )

    def _render_current_observation(self) -> GridObservation:
        return self._reset(0)

    def system_prompt(self) -> str:
        return "You play a game. Move around."

    def env_id(self) -> str:
        return "test/g-v0"


@pytest.fixture
def game():
    g = _Game()
    g.reset(0)
    return g


def test_system_prompt_contains_game_rules(game):
    sp = build_system_prompt(game, max_output_tokens=512)
    assert "You play a game." in sp
    assert "Move around." in sp


def test_system_prompt_declares_budget(game):
    sp = build_system_prompt(game, max_output_tokens=512)
    assert "512" in sp


def test_system_prompt_documents_xml_format(game):
    sp = build_system_prompt(game, max_output_tokens=512)
    # Thinking-mode chat templates (Qwen3.5) prefill <think> on the model's
    # behalf, so the system prompt no longer asks the model to emit it
    # itself. Only the <action> tag is contractually required.
    assert "<action>" in sp and "</action>" in sp


def test_system_prompt_lists_actions(game):
    sp = build_system_prompt(game, max_output_tokens=512)
    for name in ("LEFT", "RIGHT", "UP", "DOWN"):
        assert name in sp


def test_system_prompt_describes_observation_conventions(game):
    sp = build_system_prompt(game, max_output_tokens=512)
    assert "OBSERVATION CONVENTIONS" in sp
    assert "[Legend]" in sp
    assert "Step: T / N" in sp


def test_system_prompt_omits_memory_block_when_not_in_memory_mode(game):
    sp = build_system_prompt(game, max_output_tokens=512, use_memory=False)
    assert "MEMORY MODE" not in sp


def test_system_prompt_includes_memory_block_when_in_memory_mode(game):
    sp = build_system_prompt(
        game, max_output_tokens=8192,
        use_memory=True, memory_update_max_tokens=4096,
    )
    assert "MEMORY MODE" in sp
    assert "<memory>" in sp
    assert "</memory>" in sp
    # The block should also tell the agent the memory budget so it can
    # plan its memory length.
    assert "4096" in sp


def test_user_turn_zero_no_history_section(game):
    frames: deque = deque(maxlen=4)
    text = render_user_turn(
        game,
        frames,
        current_obs="[Legend]\nA — a\n\n[HUD]\nstep=0\n\n[Grid]\n.",
        turn=0,
        max_output_tokens=512,
    )
    assert "[History" not in text
    assert "[Legend]" in text
    assert "[Current Observation" in text or "[Observation" in text
    # The action spec now lives ONLY in the cached system prompt; the
    # per-turn user content just nudges the model to emit a tag now.
    assert "[Actions]" not in text
    assert "Now emit your move" in text


def test_user_turn_omits_memory_by_default(game):
    frames: deque = deque(maxlen=4)
    text = render_user_turn(
        game, frames, current_obs="[Grid]\n.", turn=0, max_output_tokens=512
    )
    assert "[Memory]" not in text


def test_user_turn_includes_memory_when_provided(game):
    frames: deque = deque(maxlen=4)
    text = render_user_turn(
        game,
        frames,
        current_obs="[Grid]\n.",
        turn=0,
        max_output_tokens=512,
        memory="visited start",
    )
    assert text.startswith("[Memory]")
    assert "<memory>\nvisited start\n</memory>" in text


def test_user_turn_includes_empty_memory_block_when_enabled(game):
    frames: deque = deque(maxlen=4)
    text = render_user_turn(
        game,
        frames,
        current_obs="[Grid]\n.",
        turn=0,
        max_output_tokens=512,
        memory="",
    )
    assert text.startswith("[Memory]")
    assert "<memory>\n\n</memory>" in text


def test_user_turn_with_history_dedups_legend(game):
    frames = deque(
        [
            ("[Legend]\nA — a\n\n[Grid]\nA.", "LEFT", 0.0),
            ("[Legend]\nA — a\n\n[Grid]\n.A", "RIGHT", 0.0),
        ],
        maxlen=4,
    )
    current = "[Legend]\nA — a\n\n[Grid]\nAA"
    text = render_user_turn(game, frames, current, turn=2, max_output_tokens=512)
    # Legend appears once globally (at top), not inside frames or current.
    assert text.count("[Legend]") == 1


def test_user_turn_history_window_respected(game):
    frames = deque(
        [(f"[Grid]\n{i}", "LEFT", float(i)) for i in range(4)],
        maxlen=4,
    )
    text = render_user_turn(
        game, frames, current_obs="[Grid]\nC", turn=4, max_output_tokens=512
    )
    # Four history entries rendered — each formatted as "chose <action> → reward ...".
    assert text.count("chose ") == 4


def test_history_renders_forfeit_with_explicit_marker():
    from collections import deque
    from glyphbench.verifiers_integration.prompting import _render_history

    frames = deque(
        [
            ("[Grid]\nA", "MOVE_FORWARD", 0.0),
            ("[Grid]\nA", "FORFEIT", 0.0),
            ("[Grid]\nA", "MOVE_FORWARD", 1.0),
        ]
    )
    text = _render_history(list(frames), current_turn=4)
    # Normal turns: "chose NAME → reward R"
    assert "chose MOVE_FORWARD → reward +0.000" in text
    # Forfeit turn gets an explicit explanation
    assert "chose FORFEIT (parse failed) → reward 0" in text


def test_user_turn_reminds_format(game):
    frames: deque = deque(maxlen=4)
    text = render_user_turn(
        game, frames, current_obs="[Grid]\n.", turn=0, max_output_tokens=512
    )
    # Per-turn reminder asks for the action tag. The full budget statement
    # lives in the (cached) system prompt; we don't echo it every turn.
    assert "<action>" in text
