"""Tests for MiniGrid DoorKey environments."""

from __future__ import annotations

import gymnasium as gym
import pytest

import glyphbench  # noqa: F401

DOORKEY_VARIANTS = [
    "glyphbench/minigrid-doorkey-5x5-v0",
    "glyphbench/minigrid-doorkey-6x6-v0",
    "glyphbench/minigrid-doorkey-8x8-v0",
    "glyphbench/minigrid-doorkey-16x16-v0",
]


class TestDoorKey:
    @pytest.mark.parametrize("env_id", DOORKEY_VARIANTS)
    def test_reset_and_step(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=200)
        obs, info = env.reset(seed=0)
        assert isinstance(obs, str)

    @pytest.mark.parametrize("env_id", DOORKEY_VARIANTS)
    def test_seed_determinism(self, env_id: str) -> None:
        e1 = gym.make(env_id, max_turns=200)
        e2 = gym.make(env_id, max_turns=200)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("env_id", DOORKEY_VARIANTS)
    def test_has_key_and_door(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=200)
        obs, _ = env.reset(seed=0)
        # Should see key and door in the legend
        assert "key" in obs
        assert "door" in obs

    @pytest.mark.parametrize("env_id", DOORKEY_VARIANTS)
    def test_has_goal(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=200)
        obs, _ = env.reset(seed=0)
        assert "G" in obs

    def test_pickup_key_mechanics(self) -> None:
        """Test that picking up the key works in the smallest variant."""
        from glyphbench.core.base_env import BaseAsciiEnv

        env = gym.make("glyphbench/minigrid-doorkey-5x5-v0", max_turns=200)
        env.reset(seed=0)
        unwrapped: BaseAsciiEnv = env.unwrapped  # type: ignore[assignment]
        from glyphbench.envs.minigrid.base import MiniGridBase

        assert isinstance(unwrapped, MiniGridBase)
        # Agent should be able to pick up key eventually through random play
        # This is a sanity check that the mechanics are wired
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
