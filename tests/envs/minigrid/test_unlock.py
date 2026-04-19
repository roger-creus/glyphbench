"""Tests for MiniGrid Unlock and UnlockPickup environments."""

from __future__ import annotations

import gymnasium as gym
import pytest

import atlas_rl  # noqa: F401

UNLOCK_VARIANTS = [
    "atlas_rl/minigrid-unlock-v0",
    "atlas_rl/minigrid-unlockpickup-v0",
]


class TestUnlock:
    @pytest.mark.parametrize("env_id", UNLOCK_VARIANTS)
    def test_reset_and_step(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=200)
        obs, info = env.reset(seed=0)
        assert isinstance(obs, str)

    @pytest.mark.parametrize("env_id", UNLOCK_VARIANTS)
    def test_seed_determinism(self, env_id: str) -> None:
        e1 = gym.make(env_id, max_turns=200)
        e2 = gym.make(env_id, max_turns=200)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("env_id", UNLOCK_VARIANTS)
    def test_has_key_and_door(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=200)
        obs, _ = env.reset(seed=0)
        assert "K" in obs
        assert "D" in obs

    def test_unlock_has_goal(self) -> None:
        env = gym.make("atlas_rl/minigrid-unlock-v0", max_turns=200)
        obs, _ = env.reset(seed=0)
        assert "G" in obs

    def test_unlockpickup_has_box(self) -> None:
        env = gym.make("atlas_rl/minigrid-unlockpickup-v0", max_turns=200)
        obs, _ = env.reset(seed=0)
        assert "B" in obs

    @pytest.mark.parametrize("env_id", UNLOCK_VARIANTS)
    def test_random_rollout_no_crash(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=200)
        env.reset(seed=0)
        for _ in range(200):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
