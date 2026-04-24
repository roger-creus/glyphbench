"""Tests for Empty room size variants and random-start variants."""

from __future__ import annotations

from glyphbench.core import make_env
import glyphbench.envs.minigrid  # register envs


import pytest

import glyphbench  # noqa: F401

EMPTY_VARIANTS = [
    ("glyphbench/minigrid-empty-6x6-v0", 8, 8),
    ("glyphbench/minigrid-empty-8x8-v0", 10, 10),
    ("glyphbench/minigrid-empty-16x16-v0", 18, 18),
]

RANDOM_VARIANTS = [
    "glyphbench/minigrid-empty-random-5x5-v0",
    "glyphbench/minigrid-empty-random-6x6-v0",
]


class TestEmptyVariants:
    @pytest.mark.parametrize("env_id,expected_w,expected_h", EMPTY_VARIANTS)
    def test_grid_size(self, env_id: str, expected_w: int, expected_h: int) -> None:
        env = make_env(env_id, max_turns=50)
        obs, _ = env.reset(0)
        grid_text = obs.split("[Grid]\n")[1].split("\n\n")[0]
        rows = grid_text.strip().split("\n")
        assert len(rows) == expected_h
        assert all(len(r) == expected_w for r in rows)

    @pytest.mark.parametrize("env_id", RANDOM_VARIANTS)
    def test_random_start_varies_across_seeds(self, env_id: str) -> None:
        positions = set()
        for seed in range(10):
            env = make_env(env_id, max_turns=50)
            _, info = env.reset(seed=seed)
            positions.add(info.get("agent_pos"))
        assert len(positions) >= 2

    @pytest.mark.parametrize("env_id", RANDOM_VARIANTS)
    def test_random_start_deterministic(self, env_id: str) -> None:
        e1 = make_env(env_id, max_turns=50)
        e2 = make_env(env_id, max_turns=50)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2
