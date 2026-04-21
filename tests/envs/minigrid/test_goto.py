"""Tests for GoToDoor and GoToObject environments."""

from __future__ import annotations

import gymnasium as gym
import pytest

import glyphbench  # noqa: F401

GOTO_VARIANTS = [
    "glyphbench/minigrid-gotodoor-5x5-v0",
    "glyphbench/minigrid-gotodoor-6x6-v0",
    "glyphbench/minigrid-gotodoor-8x8-v0",
    "glyphbench/minigrid-gotoobject-6x6-n2-v0",
]


class TestGoTo:
    @pytest.mark.parametrize("env_id", GOTO_VARIANTS)
    def test_reset_and_step(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=100)
        obs, _ = env.reset(seed=0)
        assert isinstance(obs, str)

    @pytest.mark.parametrize("env_id", GOTO_VARIANTS)
    def test_seed_determinism(self, env_id: str) -> None:
        e1 = gym.make(env_id, max_turns=100)
        e2 = gym.make(env_id, max_turns=100)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("env_id", GOTO_VARIANTS)
    def test_random_rollout_no_crash(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=100)
        for seed in range(5):
            env.reset(seed=seed)
            for _ in range(100):
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
