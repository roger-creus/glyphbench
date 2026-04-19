"""Tests for MiniGrid DistShift environments."""

from __future__ import annotations

import gymnasium as gym
import pytest

import atlas_rl  # noqa: F401

DISTSHIFT_VARIANTS = [
    "atlas_rl/minigrid-distshift1-v0",
    "atlas_rl/minigrid-distshift2-v0",
]


class TestDistShift:
    @pytest.mark.parametrize("env_id", DISTSHIFT_VARIANTS)
    def test_reset_and_step(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=100)
        obs, info = env.reset(seed=0)
        assert isinstance(obs, str)
        assert "G" in obs

    @pytest.mark.parametrize("env_id", DISTSHIFT_VARIANTS)
    def test_seed_determinism(self, env_id: str) -> None:
        e1 = gym.make(env_id, max_turns=100)
        e2 = gym.make(env_id, max_turns=100)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("env_id", DISTSHIFT_VARIANTS)
    def test_has_lava(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=100)
        obs, _ = env.reset(seed=0)
        assert "L" in obs

    @pytest.mark.parametrize("env_id", DISTSHIFT_VARIANTS)
    def test_random_rollout_no_crash(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=100)
        env.reset(seed=0)
        for _ in range(100):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
