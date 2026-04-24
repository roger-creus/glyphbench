"""Tests for Playground environment."""

from __future__ import annotations

from glyphbench.core import make_env
import glyphbench.envs.minigrid  # register envs



import glyphbench  # noqa: F401


class TestPlayground:
    def test_reset_and_step(self) -> None:
        env = make_env("glyphbench/minigrid-playground-v0", max_turns=500)
        obs, _ = env.reset(0)
        assert isinstance(obs, str)
        assert "G" in obs

    def test_seed_determinism(self) -> None:
        e1 = make_env("glyphbench/minigrid-playground-v0", max_turns=500)
        e2 = make_env("glyphbench/minigrid-playground-v0", max_turns=500)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    def test_has_diverse_objects(self) -> None:
        env = make_env("glyphbench/minigrid-playground-v0", max_turns=500)
        obs, _ = env.reset(0)
        # Should have keys, balls, and goal
        assert "key" in obs
        assert "ball" in obs
        assert "goal" in obs

    def test_grid_size(self) -> None:
        env = make_env("glyphbench/minigrid-playground-v0", max_turns=500)
        obs, _ = env.reset(0)
        grid_text = obs.split("[Grid]\n")[1].split("\n\n")[0]
        rows = grid_text.strip().split("\n")
        assert len(rows) == 18
        assert all(len(r) == 18 for r in rows)

    def test_random_rollout_no_crash(self) -> None:
        env = make_env("glyphbench/minigrid-playground-v0", max_turns=500)
        for seed in range(5):
            env.reset(seed=seed)
            for _ in range(500):
                action = int(env.rng.integers(0, env.action_spec.n))
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
