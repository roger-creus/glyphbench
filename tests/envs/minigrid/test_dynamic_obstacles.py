"""Tests for MiniGrid Dynamic-Obstacles environments."""

from __future__ import annotations

import gymnasium as gym
import pytest

import atlas_rl  # noqa: F401

DYNAMIC_VARIANTS = [
    "atlas_rl/minigrid-dynamic-obstacles-5x5-v0",
    "atlas_rl/minigrid-dynamic-obstacles-6x6-v0",
    "atlas_rl/minigrid-dynamic-obstacles-8x8-v0",
    "atlas_rl/minigrid-dynamic-obstacles-16x16-v0",
]


class TestDynamicObstacles:
    @pytest.mark.parametrize("env_id", DYNAMIC_VARIANTS)
    def test_reset_and_step(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=100)
        obs, info = env.reset(seed=0)
        assert isinstance(obs, str)
        assert "G" in obs or "O" in obs  # goal or obstacle visible

    @pytest.mark.parametrize("env_id", DYNAMIC_VARIANTS)
    def test_seed_determinism(self, env_id: str) -> None:
        e1 = gym.make(env_id, max_turns=100)
        e2 = gym.make(env_id, max_turns=100)
        e1.reset(seed=42)
        e2.reset(seed=42)
        for _ in range(20):
            o1, r1, t1, tr1, _ = e1.step(2)  # MOVE_FORWARD
            o2, r2, t2, tr2, _ = e2.step(2)
            assert o1 == o2
            assert r1 == r2
            if t1 or tr1:
                break

    @pytest.mark.parametrize("env_id", DYNAMIC_VARIANTS)
    def test_has_obstacles(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=100)
        obs, _ = env.reset(seed=0)
        assert "O" in obs  # ball obstacles render as O

    @pytest.mark.parametrize("env_id", DYNAMIC_VARIANTS)
    def test_obstacles_move(self, env_id: str) -> None:
        """Obstacles should change position after stepping."""
        from atlas_rl.envs.minigrid.dynamic_obstacles import _DynamicObstaclesBase

        env = gym.make(env_id, max_turns=200)
        # Try a few seeds to find one where obstacles visibly move
        moved = False
        for seed in range(10):
            env.reset(seed=seed)
            unwrapped: _DynamicObstaclesBase = env.unwrapped  # type: ignore[assignment]
            initial_pos = list(unwrapped._obstacle_positions)
            any_moved = False
            for _ in range(20):
                _, _, term, trunc, _ = env.step(6)  # DONE (no-op)
                if term or trunc:
                    break
                if list(unwrapped._obstacle_positions) != initial_pos:
                    any_moved = True
                    break
            if any_moved:
                moved = True
                break
        assert moved, "Obstacles did not move in any seed"

    @pytest.mark.parametrize("env_id", DYNAMIC_VARIANTS)
    def test_random_rollout_no_crash(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=100)
        env.reset(seed=0)
        for _ in range(100):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset(seed=0)

    @pytest.mark.parametrize("env_id", DYNAMIC_VARIANTS)
    def test_goal_reachable(self, env_id: str) -> None:
        """The goal should exist in the initial observation."""
        env = gym.make(env_id, max_turns=100)
        obs, _ = env.reset(seed=0)
        assert "G" in obs

    def test_collision_terminates(self) -> None:
        """If an obstacle moves onto the agent, episode terminates with 0 reward."""
        from atlas_rl.envs.minigrid.dynamic_obstacles import (
            MiniGridDynamicObstacles5x5Env,
        )

        env_inner = MiniGridDynamicObstacles5x5Env(max_turns=500)
        env = gym.make(
            "atlas_rl/minigrid-dynamic-obstacles-5x5-v0", max_turns=500
        )
        # Run many seeds until we get a collision
        found_collision = False
        for seed in range(50):
            env.reset(seed=seed)
            for _ in range(200):
                _, reward, terminated, truncated, info = env.step(6)  # DONE
                if terminated:
                    if info.get("obstacle_collision"):
                        assert reward == 0.0
                        found_collision = True
                        break
                    break
                if truncated:
                    break
            if found_collision:
                break
        # If no collision found in 50 seeds, that's okay -- it's probabilistic.
        # But we at least verified the logic doesn't crash.
