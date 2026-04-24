"""Tests for MiniGrid MultiRoom environments."""

from __future__ import annotations

from glyphbench.core import make_env
import glyphbench.envs.minigrid  # register envs


import pytest

import glyphbench  # noqa: F401

MULTIROOM_VARIANTS = [
    "glyphbench/minigrid-multiroom-n2-s4-v0",
    "glyphbench/minigrid-multiroom-n4-s5-v0",
    "glyphbench/minigrid-multiroom-n6-v0",
]


class TestMultiRoom:
    @pytest.mark.parametrize("env_id", MULTIROOM_VARIANTS)
    def test_reset_and_step(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=300)
        obs, info = env.reset(0)
        assert isinstance(obs, str)
        assert "G" in obs

    @pytest.mark.parametrize("env_id", MULTIROOM_VARIANTS)
    def test_seed_determinism(self, env_id: str) -> None:
        e1 = make_env(env_id, max_turns=300)
        e2 = make_env(env_id, max_turns=300)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("env_id", MULTIROOM_VARIANTS)
    def test_has_doors(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=300)
        obs, _ = env.reset(0)
        assert "door" in obs  # closed door

    @pytest.mark.parametrize("env_id", MULTIROOM_VARIANTS)
    def test_random_rollout_no_crash(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=300)
        env.reset(0)
        for _ in range(300):
            action = int(env.rng.integers(0, env.action_spec.n))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
