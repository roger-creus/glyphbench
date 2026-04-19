"""Tests for Memory environments."""

from __future__ import annotations

import gymnasium as gym
import pytest

import atlas_rl  # noqa: F401

MEMORY_VARIANTS = [
    "atlas_rl/minigrid-memory-s7-v0",
    "atlas_rl/minigrid-memory-s9-v0",
    "atlas_rl/minigrid-memory-s11-v0",
    "atlas_rl/minigrid-memory-s13-v0",
    "atlas_rl/minigrid-memory-s17-v0",
]


class TestMemory:
    @pytest.mark.parametrize("env_id", MEMORY_VARIANTS)
    def test_reset_and_step(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=200)
        obs, _ = env.reset(seed=0)
        assert isinstance(obs, str)
        assert "G" in obs

    @pytest.mark.parametrize("env_id", MEMORY_VARIANTS)
    def test_seed_determinism(self, env_id: str) -> None:
        e1 = gym.make(env_id, max_turns=200)
        e2 = gym.make(env_id, max_turns=200)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("env_id", MEMORY_VARIANTS)
    def test_has_memory_object(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=200)
        obs, _ = env.reset(seed=0)
        # Should have a key (K) or ball (O) as the memory object
        assert "K" in obs or "O" in obs

    @pytest.mark.parametrize("env_id", MEMORY_VARIANTS)
    def test_random_rollout_no_crash(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=200)
        for seed in range(3):
            env.reset(seed=seed)
            for _ in range(200):
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
