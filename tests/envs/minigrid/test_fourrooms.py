"""Tests for MiniGrid FourRooms environment."""

from __future__ import annotations

from glyphbench.core import make_env
import glyphbench.envs.minigrid  # register envs



import glyphbench  # noqa: F401


class TestFourRooms:
    def test_reset_and_step(self) -> None:
        env = make_env("glyphbench/minigrid-fourrooms-v0", max_turns=200)
        obs, info = env.reset(0)
        assert isinstance(obs, str)
        assert "G" in obs

    def test_seed_determinism(self) -> None:
        e1 = make_env("glyphbench/minigrid-fourrooms-v0", max_turns=200)
        e2 = make_env("glyphbench/minigrid-fourrooms-v0", max_turns=200)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    def test_grid_size(self) -> None:
        env = make_env("glyphbench/minigrid-fourrooms-v0", max_turns=200)
        obs, _ = env.reset(0)
        grid_text = obs.split("[Grid]\n")[1].split("\n\n")[0]
        rows = grid_text.strip().split("\n")
        assert len(rows) == 19
        assert all(len(r) == 19 for r in rows)

    def test_has_interior_walls(self) -> None:
        env = make_env("glyphbench/minigrid-fourrooms-v0", max_turns=200)
        obs, _ = env.reset(0)
        grid_text = obs.split("[Grid]\n")[1].split("\n\n")[0]
        rows = grid_text.strip().split("\n")
        # Row 9 should have many █ (horizontal wall)
        wall_count = rows[9].count("█")
        assert wall_count >= 10  # Most of row 9 should be wall

    def test_random_rollout_no_crash(self) -> None:
        env = make_env("glyphbench/minigrid-fourrooms-v0", max_turns=200)
        env.reset(0)
        for _ in range(200):
            action = int(env.rng.integers(0, env.action_spec.n))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
