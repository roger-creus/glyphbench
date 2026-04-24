"""Tests for MiniGrid Crossing and SimpleCrossing environments."""

from __future__ import annotations

from glyphbench.core import make_env
import glyphbench.envs.minigrid  # register envs


import pytest

import glyphbench  # noqa: F401

LAVA_CROSSING = [
    "glyphbench/minigrid-crossing-n1-v0",
    "glyphbench/minigrid-crossing-n2-v0",
    "glyphbench/minigrid-crossing-n3-v0",
]

SAFE_CROSSING = [
    "glyphbench/minigrid-crossing-n1-safe-v0",
    "glyphbench/minigrid-crossing-n2-safe-v0",
    "glyphbench/minigrid-crossing-n3-safe-v0",
]

SIMPLE_CROSSING = [
    "glyphbench/minigrid-simplecrossing-n1-v0",
    "glyphbench/minigrid-simplecrossing-n2-v0",
    "glyphbench/minigrid-simplecrossing-n3-v0",
    "glyphbench/minigrid-simplecrossing-easy-n1-v0",
    "glyphbench/minigrid-simplecrossing-easy-n2-v0",
    "glyphbench/minigrid-simplecrossing-easy-n3-v0",
]

ALL_CROSSING = LAVA_CROSSING + SAFE_CROSSING + SIMPLE_CROSSING


class TestCrossing:
    @pytest.mark.parametrize("env_id", ALL_CROSSING)
    def test_reset_and_step(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=100)
        obs, _ = env.reset(0)
        assert isinstance(obs, str)
        assert "G" in obs

    @pytest.mark.parametrize("env_id", ALL_CROSSING)
    def test_seed_determinism(self, env_id: str) -> None:
        e1 = make_env(env_id, max_turns=100)
        e2 = make_env(env_id, max_turns=100)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("env_id", LAVA_CROSSING)
    def test_lava_crossing_has_lava(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=100)
        obs, _ = env.reset(0)
        assert "L" in obs

    @pytest.mark.parametrize("env_id", SAFE_CROSSING)
    def test_safe_crossing_has_water(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=100)
        obs, _ = env.reset(0)
        assert "≈" in obs

    @pytest.mark.parametrize("env_id", ALL_CROSSING)
    def test_random_rollout_no_crash(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=100)
        env.reset(0)
        for _ in range(100):
            action = int(env.rng.integers(0, env.action_spec.n))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
