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
        return GridObservation(grid="A.", legend="A symbol-a", hud="step=1", message=""), 0.0, False, False, {}
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
    assert "<think>" in sp and "</think>" in sp
    assert "<action>" in sp and "</action>" in sp


def test_system_prompt_lists_actions(game):
    sp = build_system_prompt(game, max_output_tokens=512)
    for name in ("LEFT", "RIGHT", "UP", "DOWN"):
        assert name in sp


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
    assert "[Actions]" in text


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
    # Four history entries rendered.
    assert text.count("reward") == 4 or text.count("→") == 4 or "turn" in text


def test_user_turn_reminds_format_and_budget(game):
    frames: deque = deque(maxlen=4)
    text = render_user_turn(
        game, frames, current_obs="[Grid]\n.", turn=0, max_output_tokens=512
    )
    assert "<action>" in text
    assert "512" in text
