"""Tests for MiniGrid Crossing and SimpleCrossing environments."""

from __future__ import annotations

import gymnasium as gym
import pytest

import atlas_rl  # noqa: F401

LAVA_CROSSING = [
    "atlas_rl/minigrid-crossing-n1-v0",
    "atlas_rl/minigrid-crossing-n2-v0",
    "atlas_rl/minigrid-crossing-n3-v0",
]

SAFE_CROSSING = [
    "atlas_rl/minigrid-crossing-n1-safe-v0",
    "atlas_rl/minigrid-crossing-n2-safe-v0",
    "atlas_rl/minigrid-crossing-n3-safe-v0",
]

SIMPLE_CROSSING = [
    "atlas_rl/minigrid-simplecrossing-n1-v0",
    "atlas_rl/minigrid-simplecrossing-n2-v0",
    "atlas_rl/minigrid-simplecrossing-n3-v0",
    "atlas_rl/minigrid-simplecrossing-easy-n1-v0",
    "atlas_rl/minigrid-simplecrossing-easy-n2-v0",
    "atlas_rl/minigrid-simplecrossing-easy-n3-v0",
]

ALL_CROSSING = LAVA_CROSSING + SAFE_CROSSING + SIMPLE_CROSSING


class TestCrossing:
    @pytest.mark.parametrize("env_id", ALL_CROSSING)
    def test_reset_and_step(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=100)
        obs, _ = env.reset(seed=0)
        assert isinstance(obs, str)
        assert "G" in obs

    @pytest.mark.parametrize("env_id", ALL_CROSSING)
    def test_seed_determinism(self, env_id: str) -> None:
        e1 = gym.make(env_id, max_turns=100)
        e2 = gym.make(env_id, max_turns=100)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("env_id", LAVA_CROSSING)
    def test_lava_crossing_has_lava(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=100)
        obs, _ = env.reset(seed=0)
        assert "L" in obs

    @pytest.mark.parametrize("env_id", SAFE_CROSSING)
    def test_safe_crossing_has_water(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=100)
        obs, _ = env.reset(seed=0)
        assert "~" in obs

    @pytest.mark.parametrize("env_id", ALL_CROSSING)
    def test_random_rollout_no_crash(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=100)
        env.reset(seed=0)
        for _ in range(100):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
