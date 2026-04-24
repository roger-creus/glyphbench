"""Tests for MiniGrid Unlock and UnlockPickup environments."""

from __future__ import annotations

from glyphbench.core import make_env
import glyphbench.envs.minigrid  # register envs


import pytest

import glyphbench  # noqa: F401

UNLOCK_VARIANTS = [
    "glyphbench/minigrid-unlock-v0",
    "glyphbench/minigrid-unlockpickup-v0",
]


class TestUnlock:
    @pytest.mark.parametrize("env_id", UNLOCK_VARIANTS)
    def test_reset_and_step(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=200)
        obs, info = env.reset(0)
        assert isinstance(obs, str)

    @pytest.mark.parametrize("env_id", UNLOCK_VARIANTS)
    def test_seed_determinism(self, env_id: str) -> None:
        e1 = make_env(env_id, max_turns=200)
        e2 = make_env(env_id, max_turns=200)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("env_id", UNLOCK_VARIANTS)
    def test_has_key_and_door(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=200)
        obs, _ = env.reset(0)
        assert "key" in obs
        assert "door" in obs

    def test_unlock_has_goal(self) -> None:
        env = make_env("glyphbench/minigrid-unlock-v0", max_turns=200)
        obs, _ = env.reset(0)
        assert "G" in obs

    def test_unlockpickup_has_box(self) -> None:
        env = make_env("glyphbench/minigrid-unlockpickup-v0", max_turns=200)
        obs, _ = env.reset(0)
        assert "box" in obs

    @pytest.mark.parametrize("env_id", UNLOCK_VARIANTS)
    def test_random_rollout_no_crash(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=200)
        env.reset(0)
        for _ in range(200):
            action = int(env.rng.integers(0, env.action_spec.n))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
