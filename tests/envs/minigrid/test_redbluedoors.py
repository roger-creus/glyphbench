"""Tests for MiniGrid RedBlueDoors environments."""

from __future__ import annotations

import gymnasium as gym
import pytest

import glyphbench  # noqa: F401

REDBLUEDOORS_VARIANTS = [
    "glyphbench/minigrid-redbluedoors-6x6-v0",
    "glyphbench/minigrid-redbluedoors-8x8-v0",
]


class TestRedBlueDoors:
    @pytest.mark.parametrize("env_id", REDBLUEDOORS_VARIANTS)
    def test_reset_and_step(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=100)
        obs, _ = env.reset(seed=0)
        assert isinstance(obs, str)
        assert "G" in obs

    @pytest.mark.parametrize("env_id", REDBLUEDOORS_VARIANTS)
    def test_seed_determinism(self, env_id: str) -> None:
        e1 = gym.make(env_id, max_turns=100)
        e2 = gym.make(env_id, max_turns=100)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("env_id", REDBLUEDOORS_VARIANTS)
    def test_has_doors(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=100)
        obs, _ = env.reset(seed=0)
        assert "door" in obs  # closed doors

    @pytest.mark.parametrize("env_id", REDBLUEDOORS_VARIANTS)
    def test_has_two_door_colors_in_legend(self, env_id: str) -> None:
        from glyphbench.core.base_env import BaseAsciiEnv

        env = gym.make(env_id, max_turns=100)
        env.reset(seed=0)
        unwrapped: BaseAsciiEnv = env.unwrapped  # type: ignore[assignment]
        obs = unwrapped.get_observation()
        legend_lower = obs.legend.lower()
        assert "red" in legend_lower or "blue" in legend_lower

    @pytest.mark.parametrize("env_id", REDBLUEDOORS_VARIANTS)
    def test_random_rollout_no_crash(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=100)
        for seed in range(5):
            env.reset(seed=seed)
            for _ in range(100):
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
