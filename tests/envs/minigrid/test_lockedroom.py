"""Tests for MiniGrid LockedRoom and BlockedUnlockPickup environments."""

from __future__ import annotations

import gymnasium as gym
import pytest

import atlas_rl  # noqa: F401

LOCKED_VARIANTS = [
    "atlas_rl/minigrid-lockedroom-v0",
    "atlas_rl/minigrid-blockedunlockpickup-v0",
]


class TestLockedRoom:
    @pytest.mark.parametrize("env_id", LOCKED_VARIANTS)
    def test_reset_and_step(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=300)
        obs, info = env.reset(seed=0)
        assert isinstance(obs, str)

    @pytest.mark.parametrize("env_id", LOCKED_VARIANTS)
    def test_seed_determinism(self, env_id: str) -> None:
        e1 = gym.make(env_id, max_turns=300)
        e2 = gym.make(env_id, max_turns=300)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("env_id", LOCKED_VARIANTS)
    def test_has_key_and_door(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=300)
        obs, _ = env.reset(seed=0)
        assert "key" in obs
        assert "door" in obs

    def test_lockedroom_has_goal(self) -> None:
        env = gym.make("atlas_rl/minigrid-lockedroom-v0", max_turns=300)
        obs, _ = env.reset(seed=0)
        assert "G" in obs

    def test_blockedunlockpickup_has_ball(self) -> None:
        env = gym.make("atlas_rl/minigrid-blockedunlockpickup-v0", max_turns=300)
        obs, _ = env.reset(seed=0)
        assert "ball" in obs  # ball in legend

    @pytest.mark.parametrize("env_id", LOCKED_VARIANTS)
    def test_random_rollout_no_crash(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=300)
        for seed in range(5):
            env.reset(seed=seed)
            for _ in range(300):
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
