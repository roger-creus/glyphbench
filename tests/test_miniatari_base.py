"""Tests for MiniatariBase reward helpers and lifecycle."""
from __future__ import annotations

from glyphbench.core.action import ActionSpec
from glyphbench.envs.miniatari.base import MiniatariBase


class _FakeMini(MiniatariBase):
    action_spec = ActionSpec(names=("NOOP",), descriptions=("do nothing",))

    def env_id(self) -> str:
        return "glyphbench/__test-mini-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(4, 4)

    def _game_step(self, action_name: str):
        return 0.0, False, {}


def test_progress_reward_bounded_to_one():
    env = _FakeMini()
    assert env._progress_reward(5) == 0.2
    assert env._progress_reward(1) == 1.0


def test_agent_score_reward():
    env = _FakeMini()
    assert env._agent_score_reward(3) == 1.0 / 3


def test_opp_score_reward():
    env = _FakeMini()
    assert env._opp_score_reward(3) == -1.0 / 3


def test_death_reward_is_negative_one():
    env = _FakeMini()
    assert env._death_reward() == -1.0


def test_milestone_reward_passes_through():
    env = _FakeMini()
    assert env._milestone_reward(0.2) == 0.2
    assert env._milestone_reward(0.5) == 0.5


def test_default_max_turns_is_200():
    env = _FakeMini()
    assert env.max_turns == 200


def test_explicit_max_turns_override():
    env = _FakeMini(max_turns=300)
    assert env.max_turns == 300
