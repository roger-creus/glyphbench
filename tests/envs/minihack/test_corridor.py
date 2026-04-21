"""Tests for MiniHack Corridor environments."""

from __future__ import annotations

import pytest

from glyphbench.envs.minihack.corridor import (
    MiniHackCorridorR2Env,
    MiniHackCorridorR3Env,
    MiniHackCorridorR5Env,
)

CORRIDOR_CLASSES = [MiniHackCorridorR2Env, MiniHackCorridorR3Env, MiniHackCorridorR5Env]
CORRIDOR_IDS = [
    "glyphbench/minihack-corridor-r2-v0",
    "glyphbench/minihack-corridor-r3-v0",
    "glyphbench/minihack-corridor-r5-v0",
]


class TestCorridorEnvs:
    """Tests for all Corridor variants."""

    @pytest.mark.parametrize(
        "cls,expected_id", zip(CORRIDOR_CLASSES, CORRIDOR_IDS, strict=True)
    )
    def test_env_id(self, cls: type, expected_id: str) -> None:
        env = cls()
        assert env.env_id() == expected_id

    @pytest.mark.parametrize("cls", CORRIDOR_CLASSES)
    def test_reset_determinism(self, cls: type) -> None:
        e1, e2 = cls(), cls()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("cls", CORRIDOR_CLASSES)
    def test_reset_produces_valid_grid(self, cls: type) -> None:
        env = cls()
        obs_str, _ = env.reset(seed=0)
        assert "@" in obs_str
        assert "⇣" in obs_str
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str
        assert "[HUD]" in obs_str

    @pytest.mark.parametrize("cls", CORRIDOR_CLASSES)
    def test_grid_rows_same_length(self, cls: type) -> None:
        env = cls()
        env.reset(seed=0)
        grid_obs = env.get_observation()
        lines = grid_obs.grid.split("\n")
        lengths = [len(line) for line in lines]
        assert len(set(lengths)) == 1, f"Unequal row lengths: {lengths}"

    @pytest.mark.parametrize("cls", CORRIDOR_CLASSES)
    def test_player_can_reach_goal(self, cls: type) -> None:
        """Navigate east repeatedly; should eventually reach the goal."""
        env = cls(max_turns=500)
        env.reset(seed=0)
        move_e = env.action_spec.index_of("MOVE_E")
        reached = False
        for _ in range(500):
            _, reward, terminated, truncated, info = env.step(move_e)
            if terminated:
                assert reward == 1.0
                assert info.get("goal_reached")
                reached = True
                break
            if truncated:
                break
        assert reached, "Agent should reach goal by moving east through corridors"

    @pytest.mark.parametrize("cls", CORRIDOR_CLASSES)
    def test_walls_block_movement(self, cls: type) -> None:
        env = cls()
        env.reset(seed=0)
        # Move north repeatedly into walls
        move_n = env.action_spec.index_of("MOVE_N")
        for _ in range(20):
            env.step(move_n)
        # Player should be blocked at some point (can't go past wall)
        assert env._player_pos[1] >= 1  # can't go past top border

    def test_r2_has_2_rooms(self) -> None:
        env = MiniHackCorridorR2Env()
        assert env._num_rooms == 2

    def test_r3_has_3_rooms(self) -> None:
        env = MiniHackCorridorR3Env()
        assert env._num_rooms == 3

    def test_r5_has_5_rooms(self) -> None:
        env = MiniHackCorridorR5Env()
        assert env._num_rooms == 5

    @pytest.mark.parametrize("cls", CORRIDOR_CLASSES)
    def test_action_spec(self, cls: type) -> None:
        env = cls()
        assert env.action_spec.n == 22
        assert env.noop_action_name == "WAIT"

    @pytest.mark.parametrize("cls", CORRIDOR_CLASSES)
    def test_max_turns_truncation(self, cls: type) -> None:
        env = cls(max_turns=3)
        env.reset(seed=0)
        wait = env.action_spec.index_of("WAIT")
        for i in range(3):
            _, _, terminated, truncated, _ = env.step(wait)
            if i < 2:
                assert not truncated
            else:
                assert truncated
                assert not terminated

    @pytest.mark.parametrize("cls", CORRIDOR_CLASSES)
    def test_system_prompt(self, cls: type) -> None:
        env = cls()
        prompt = env.system_prompt()
        assert len(prompt) > 50
        assert "MOVE_N" in prompt
