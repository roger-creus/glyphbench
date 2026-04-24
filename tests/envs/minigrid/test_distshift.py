"""Tests for MiniGrid DistShift environments."""

from __future__ import annotations

from glyphbench.core import make_env
import glyphbench.envs.minigrid  # register envs


import pytest

import glyphbench  # noqa: F401

DISTSHIFT_VARIANTS = [
    "glyphbench/minigrid-distshift1-v0",
    "glyphbench/minigrid-distshift2-v0",
]


class TestDistShift:
    @pytest.mark.parametrize("env_id", DISTSHIFT_VARIANTS)
    def test_reset_and_step(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=100)
        obs, info = env.reset(0)
        assert isinstance(obs, str)
        assert "G" in obs

    @pytest.mark.parametrize("env_id", DISTSHIFT_VARIANTS)
    def test_seed_determinism(self, env_id: str) -> None:
        e1 = make_env(env_id, max_turns=100)
        e2 = make_env(env_id, max_turns=100)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("env_id", DISTSHIFT_VARIANTS)
    def test_has_lava(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=100)
        obs, _ = env.reset(0)
        assert "L" in obs

    @pytest.mark.parametrize("env_id", DISTSHIFT_VARIANTS)
    def test_random_rollout_no_crash(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=100)
        env.reset(0)
        for _ in range(100):
            action = int(env.rng.integers(0, env.action_spec.n))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
