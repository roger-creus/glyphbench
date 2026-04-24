"""Tests for MiniGrid Fetch and PutNear environments."""

from __future__ import annotations

from glyphbench.core import make_env
import glyphbench.envs.minigrid  # register envs


import pytest

import glyphbench  # noqa: F401

FETCH_VARIANTS = [
    "glyphbench/minigrid-fetch-5x5-n2-v0",
    "glyphbench/minigrid-fetch-6x6-n2-v0",
    "glyphbench/minigrid-fetch-8x8-n3-v0",
]

PUTNEAR_VARIANTS = [
    "glyphbench/minigrid-putnear-6x6-n2-v0",
    "glyphbench/minigrid-putnear-8x8-n3-v0",
]

ALL_VARIANTS = FETCH_VARIANTS + PUTNEAR_VARIANTS


class TestFetchPutNear:
    @pytest.mark.parametrize("env_id", ALL_VARIANTS)
    def test_reset_and_step(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=100)
        obs, _ = env.reset(0)
        assert isinstance(obs, str)

    @pytest.mark.parametrize("env_id", ALL_VARIANTS)
    def test_seed_determinism(self, env_id: str) -> None:
        e1 = make_env(env_id, max_turns=100)
        e2 = make_env(env_id, max_turns=100)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("env_id", ALL_VARIANTS)
    def test_has_objects(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=100)
        obs, _ = env.reset(0)
        # Should have at least one pickable object (key or ball)
        assert "key" in obs or "ball" in obs

    @pytest.mark.parametrize("env_id", ALL_VARIANTS)
    def test_random_rollout_no_crash(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=100)
        for seed in range(5):
            env.reset(seed=seed)
            for _ in range(100):
                action = int(env.rng.integers(0, env.action_spec.n))
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
