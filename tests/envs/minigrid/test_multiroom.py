"""Tests for MiniGrid MultiRoom environments."""

from __future__ import annotations

import gymnasium as gym
import pytest

import atlas_rl  # noqa: F401

MULTIROOM_VARIANTS = [
    "atlas_rl/minigrid-multiroom-n2-s4-v0",
    "atlas_rl/minigrid-multiroom-n4-s5-v0",
    "atlas_rl/minigrid-multiroom-n6-v0",
]


class TestMultiRoom:
    @pytest.mark.parametrize("env_id", MULTIROOM_VARIANTS)
    def test_reset_and_step(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=300)
        obs, info = env.reset(seed=0)
        assert isinstance(obs, str)
        assert "G" in obs

    @pytest.mark.parametrize("env_id", MULTIROOM_VARIANTS)
    def test_seed_determinism(self, env_id: str) -> None:
        e1 = gym.make(env_id, max_turns=300)
        e2 = gym.make(env_id, max_turns=300)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("env_id", MULTIROOM_VARIANTS)
    def test_has_doors(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=300)
        obs, _ = env.reset(seed=0)
        assert "D" in obs  # closed door

    @pytest.mark.parametrize("env_id", MULTIROOM_VARIANTS)
    def test_random_rollout_no_crash(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=300)
        env.reset(seed=0)
        for _ in range(300):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
