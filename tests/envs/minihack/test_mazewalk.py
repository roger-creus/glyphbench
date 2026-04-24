"""Tests for MiniHack MazeWalk environments."""

from __future__ import annotations

import pytest

from glyphbench.envs.minihack.mazewalk import (
    MiniHackMazeWalk9x9Env,
    MiniHackMazeWalk15x15Env,
    MiniHackMazeWalk45x19Env,
    MiniHackMazeWalkMapped9x9Env,
    MiniHackMazeWalkMapped15x15Env,
    MiniHackMazeWalkMapped45x19Env,
)

DARK_CLASSES = [
    MiniHackMazeWalk9x9Env,
    MiniHackMazeWalk15x15Env,
    MiniHackMazeWalk45x19Env,
]
MAPPED_CLASSES = [
    MiniHackMazeWalkMapped9x9Env,
    MiniHackMazeWalkMapped15x15Env,
    MiniHackMazeWalkMapped45x19Env,
]
ALL_CLASSES = DARK_CLASSES + MAPPED_CLASSES

DARK_IDS = [
    "glyphbench/minihack-mazewalk-9x9-v0",
    "glyphbench/minihack-mazewalk-15x15-v0",
    "glyphbench/minihack-mazewalk-45x19-v0",
]
MAPPED_IDS = [
    "glyphbench/minihack-mazewalk-mapped-9x9-v0",
    "glyphbench/minihack-mazewalk-mapped-15x15-v0",
    "glyphbench/minihack-mazewalk-mapped-45x19-v0",
]
ALL_IDS = DARK_IDS + MAPPED_IDS


class TestMazeWalkEnvs:
    """Tests for all MazeWalk variants."""

    @pytest.mark.parametrize("cls,expected_id", zip(ALL_CLASSES, ALL_IDS, strict=True))
    def test_env_id(self, cls: type, expected_id: str) -> None:
        env = cls()
        assert env.env_id() == expected_id

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_reset_determinism(self, cls: type) -> None:
        e1, e2 = cls(), cls()
        o1, _ = e1.reset(42)
        o2, _ = e2.reset(42)
        assert o1 == o2

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_reset_produces_valid_grid(self, cls: type) -> None:
        env = cls()
        obs_str, _ = env.reset(0)
        assert "@" in obs_str
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_grid_rows_same_length(self, cls: type) -> None:
        env = cls()
        env.reset(0)
        grid_obs = env.get_observation()
        lines = grid_obs.grid.split("\n")
        lengths = [len(line) for line in lines]
        assert len(set(lengths)) == 1, f"Unequal row lengths: {lengths}"

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_maze_has_walls_and_floor(self, cls: type) -> None:
        """Maze should have both walls (█) and floor (·) cells."""
        env = cls()
        env.reset(0)
        flat = "".join("".join(row) for row in env._grid)
        assert "█" in flat, "Maze should have wall cells"
        assert "·" in flat or "⇣" in flat, "Maze should have floor/goal cells"

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_player_at_1_1(self, cls: type) -> None:
        env = cls()
        env.reset(0)
        assert env._player_pos == (1, 1)

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_stairs_exist(self, cls: type) -> None:
        env = cls()
        env.reset(0)
        assert env._goal_pos is not None
        gx, gy = env._goal_pos
        assert env._grid[gy][gx] == "⇣"

    @pytest.mark.parametrize("cls", DARK_CLASSES)
    def test_dark_variant_is_dark(self, cls: type) -> None:
        env = cls()
        env.reset(0)
        assert env._dark is True

    @pytest.mark.parametrize("cls", MAPPED_CLASSES)
    def test_mapped_variant_is_not_dark(self, cls: type) -> None:
        env = cls()
        env.reset(0)
        assert env._dark is False

    def test_maze_is_solvable_9x9(self) -> None:
        """Use BFS to verify the maze is solvable (path exists from player to goal)."""
        env = MiniHackMazeWalkMapped9x9Env(max_turns=500)
        env.reset(0)
        assert self._is_solvable(env)

    def test_maze_is_solvable_15x15(self) -> None:
        env = MiniHackMazeWalkMapped15x15Env(max_turns=1000)
        env.reset(0)
        assert self._is_solvable(env)

    def test_maze_is_solvable_45x19(self) -> None:
        env = MiniHackMazeWalkMapped45x19Env(max_turns=2000)
        env.reset(0)
        assert self._is_solvable(env)

    def test_different_seeds_different_mazes(self) -> None:
        """Different seeds should produce different mazes."""
        e1 = MiniHackMazeWalkMapped9x9Env()
        e2 = MiniHackMazeWalkMapped9x9Env()
        e1.reset(0)
        e2.reset(1)
        # Compare internal grids
        g1 = tuple(tuple(row) for row in e1._grid)
        g2 = tuple(tuple(row) for row in e2._grid)
        assert g1 != g2, "Different seeds should produce different mazes"

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_action_spec(self, cls: type) -> None:
        env = cls()
        assert env.action_spec.n == 22
        assert env.noop_action_name == "WAIT"

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_max_turns_truncation(self, cls: type) -> None:
        env = cls(max_turns=3)
        env.reset(0)
        wait = env.action_spec.index_of("WAIT")
        for i in range(3):
            _, _, terminated, truncated, _ = env.step(wait)
            if i < 2:
                assert not truncated
            else:
                assert truncated
                assert not terminated

    @pytest.mark.parametrize("cls", ALL_CLASSES)
    def test_system_prompt(self, cls: type) -> None:
        env = cls()
        prompt = env.system_prompt()
        assert len(prompt) > 50

    @pytest.mark.parametrize(
        "cls,expected_dims",
        [
            (MiniHackMazeWalk9x9Env, (9, 9)),
            (MiniHackMazeWalk15x15Env, (15, 15)),
            (MiniHackMazeWalk45x19Env, (45, 19)),
        ],
    )
    def test_maze_dimensions(self, cls: type, expected_dims: tuple[int, int]) -> None:
        env = cls()
        env.reset(0)
        assert env._grid_w == expected_dims[0]
        assert env._grid_h == expected_dims[1]

    @staticmethod
    def _is_solvable(env: MiniHackMazeWalkMapped9x9Env) -> bool:
        """BFS from player to goal on the internal grid."""
        from collections import deque

        px, py = env._player_pos
        assert env._goal_pos is not None
        gx, gy = env._goal_pos
        visited = set()
        queue: deque[tuple[int, int]] = deque()
        queue.append((px, py))
        visited.add((px, py))
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) == (gx, gy):
                return True
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited and env._is_walkable(nx, ny):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False
