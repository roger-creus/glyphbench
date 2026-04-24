"""Tests for ObstructedMaze environments."""

from __future__ import annotations

from glyphbench.core import make_env
import glyphbench.envs.minigrid  # register envs


import pytest

import glyphbench  # noqa: F401

OBSTRUCTEDMAZE_VARIANTS = [
    "glyphbench/minigrid-obstructedmaze-1dl-v0",
    "glyphbench/minigrid-obstructedmaze-1dlh-v0",
    "glyphbench/minigrid-obstructedmaze-1dlhb-v0",
    "glyphbench/minigrid-obstructedmaze-2dl-v0",
    "glyphbench/minigrid-obstructedmaze-2dlh-v0",
    "glyphbench/minigrid-obstructedmaze-2dlhb-v0",
    "glyphbench/minigrid-obstructedmaze-1q-v0",
    "glyphbench/minigrid-obstructedmaze-2q-v0",
    "glyphbench/minigrid-obstructedmaze-full-v0",
]


class TestObstructedMaze:
    @pytest.mark.parametrize("env_id", OBSTRUCTEDMAZE_VARIANTS)
    def test_reset_and_step(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=300)
        obs, _ = env.reset(0)
        assert isinstance(obs, str)

    @pytest.mark.parametrize("env_id", OBSTRUCTEDMAZE_VARIANTS)
    def test_seed_determinism(self, env_id: str) -> None:
        e1 = make_env(env_id, max_turns=300)
        e2 = make_env(env_id, max_turns=300)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("env_id", OBSTRUCTEDMAZE_VARIANTS)
    def test_has_doors_and_key(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=300)
        obs, _ = env.reset(0)
        assert "door" in obs
        assert "key" in obs

    @pytest.mark.parametrize("env_id", OBSTRUCTEDMAZE_VARIANTS)
    def test_random_rollout_no_crash(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=300)
        for seed in range(3):
            env.reset(seed=seed)
            for _ in range(300):
                action = int(env.rng.integers(0, env.action_spec.n))
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
