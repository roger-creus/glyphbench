"""Tests for Playground environment."""

from __future__ import annotations

import gymnasium as gym

import atlas_rl  # noqa: F401


class TestPlayground:
    def test_reset_and_step(self) -> None:
        env = gym.make("atlas_rl/minigrid-playground-v0", max_turns=500)
        obs, _ = env.reset(seed=0)
        assert isinstance(obs, str)
        assert "G" in obs

    def test_seed_determinism(self) -> None:
        e1 = gym.make("atlas_rl/minigrid-playground-v0", max_turns=500)
        e2 = gym.make("atlas_rl/minigrid-playground-v0", max_turns=500)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    def test_has_diverse_objects(self) -> None:
        env = gym.make("atlas_rl/minigrid-playground-v0", max_turns=500)
        obs, _ = env.reset(seed=0)
        # Should have keys, balls, and goal
        assert "K" in obs
        assert "O" in obs
        assert "G" in obs

    def test_grid_size(self) -> None:
        env = gym.make("atlas_rl/minigrid-playground-v0", max_turns=500)
        obs, _ = env.reset(seed=0)
        grid_text = obs.split("[Grid]\n")[1].split("\n\n")[0]
        rows = grid_text.strip().split("\n")
        assert len(rows) == 18
        assert all(len(r) == 18 for r in rows)

    def test_random_rollout_no_crash(self) -> None:
        env = gym.make("atlas_rl/minigrid-playground-v0", max_turns=500)
        for seed in range(5):
            env.reset(seed=seed)
            for _ in range(500):
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
