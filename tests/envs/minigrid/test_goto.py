"""Tests for GoToDoor and GoToObject environments."""

from __future__ import annotations

from glyphbench.core import make_env
import glyphbench.envs.minigrid  # register envs


import pytest

import glyphbench  # noqa: F401

GOTO_VARIANTS = [
    "glyphbench/minigrid-gotodoor-5x5-v0",
    "glyphbench/minigrid-gotodoor-6x6-v0",
    "glyphbench/minigrid-gotodoor-8x8-v0",
    "glyphbench/minigrid-gotoobject-6x6-n2-v0",
]


class TestGoTo:
    @pytest.mark.parametrize("env_id", GOTO_VARIANTS)
    def test_reset_and_step(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=100)
        obs, _ = env.reset(0)
        assert isinstance(obs, str)

    @pytest.mark.parametrize("env_id", GOTO_VARIANTS)
    def test_seed_determinism(self, env_id: str) -> None:
        e1 = make_env(env_id, max_turns=100)
        e2 = make_env(env_id, max_turns=100)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("env_id", GOTO_VARIANTS)
    def test_random_rollout_no_crash(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=100)
        for seed in range(5):
            env.reset(seed=seed)
            for _ in range(100):
                action = int(env.rng.integers(0, env.action_spec.n))
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
