"""Procgen Heist environment.

Maze with colored keys and locked doors. Agent must collect keys to open
matching doors and reach the goal.

Gym ID: atlas_rl/procgen-heist-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec
from atlas_rl.envs.procgen.base import ProcgenBase

# Key/door color pairs: (key_char, door_char, color_name)
_COLORS: list[tuple[str, str, str]] = [
    ("r", "R", "red"),
    ("b", "B", "blue"),
    ("y", "Y", "yellow"),
]


class HeistEnv(ProcgenBase):
    """Procgen Heist: maze with colored keys and locked doors."""

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "move up",
            "move down",
        ),
    )

    MAZE_W = 21
    MAZE_H = 21

    def env_id(self) -> str:
        return "atlas_rl/procgen-heist-v0"

    def _generate_level(self, seed: int) -> None:
        w, h = self.MAZE_W, self.MAZE_H
        self._init_world(w, h, fill="#")
        self._gen_maze(w, h)

        # Agent at top-left
        self._agent_x, self._agent_y = 1, 1

        # Collect all open floor cells (excluding agent start)
        floor_cells: list[tuple[int, int]] = []
        for yy in range(1, h - 1):
            for xx in range(1, w - 1):
                if self._world_at(xx, yy) == "." and (xx, yy) != (1, 1):
                    floor_cells.append((xx, yy))
        self.rng.shuffle(floor_cells)

        # Place goal
        self._goal_x, self._goal_y = floor_cells.pop()
        self._set_cell(self._goal_x, self._goal_y, "G")

        # Place 3 keys and 3 doors
        self._keys_held: set[str] = set()
        for key_ch, door_ch, _color in _COLORS:
            if len(floor_cells) < 2:
                break
            # Place key on a floor cell
            kx, ky = floor_cells.pop()
            self._set_cell(kx, ky, key_ch)
            # Place door: find a corridor cell (has exactly 2 floor neighbors)
            door_placed = False
            for i, (dx, dy) in enumerate(floor_cells):
                adj_floor = 0
                for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if self._world_at(dx + ddx, dy + ddy) in (".", "G", "r", "b", "y"):
                        adj_floor += 1
                if adj_floor == 2:
                    self._set_cell(dx, dy, door_ch)
                    floor_cells.pop(i)
                    door_placed = True
                    break
            if not door_placed and floor_cells:
                dx, dy = floor_cells.pop()
                self._set_cell(dx, dy, door_ch)

    def _gen_maze(self, w: int, h: int) -> None:
        """Carve a maze using recursive backtracking."""
        self._set_cell(1, 1, ".")
        stack = [(1, 1)]
        while stack:
            cx, cy = stack[-1]
            neighbors: list[tuple[int, int, int, int]] = []
            for ddx, ddy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + ddx, cy + ddy
                if (
                    1 <= nx < w - 1
                    and 1 <= ny < h - 1
                    and self._world_at(nx, ny) == "#"
                ):
                    neighbors.append((nx, ny, cx + ddx // 2, cy + ddy // 2))
            if neighbors:
                idx = int(self.rng.integers(0, len(neighbors)))
                nx, ny, wx, wy = neighbors[idx]
                self._set_cell(wx, wy, ".")
                self._set_cell(nx, ny, ".")
                stack.append((nx, ny))
            else:
                stack.pop()

    def _is_solid(self, x: int, y: int) -> bool:
        ch = self._world_at(x, y)
        # Walls and locked doors are solid
        if ch == "#":
            return True
        if ch in ("R", "B", "Y"):
            # Door is solid unless we have the matching key
            color_map = {"R": "r", "B": "b", "Y": "y"}
            return color_map[ch] not in self._keys_held
        return False

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        if action_name == "LEFT":
            self._try_move(-1, 0)
        elif action_name == "RIGHT":
            self._try_move(1, 0)
        elif action_name == "UP":
            self._try_move(0, -1)
        elif action_name == "DOWN":
            self._try_move(0, 1)

        # Check for key pickup
        cell = self._world_at(self._agent_x, self._agent_y)
        if cell in ("r", "b", "y"):
            self._keys_held.add(cell)
            self._set_cell(self._agent_x, self._agent_y, ".")
            color_names = {"r": "red", "b": "blue", "y": "yellow"}
            self._message = f"Picked up {color_names[cell]} key!"

        # Check for door opening (agent walks onto door cell)
        if cell in ("R", "B", "Y"):
            # If we got here, the door was opened via _is_solid check
            self._set_cell(self._agent_x, self._agent_y, ".")
            color_names = {"R": "red", "B": "blue", "Y": "yellow"}
            self._message = f"Opened {color_names[cell]} door!"

        # Check goal
        if self._agent_x == self._goal_x and self._agent_y == self._goal_y:
            reward = 10.0
            terminated = True
            self._message = "You reached the goal!"

        info["keys_held"] = list(self._keys_held)
        return reward, terminated, info

    def _task_description(self) -> str:
        return (
            "Navigate the maze, collect colored keys (r/b/y) to open matching "
            "doors (R/B/Y), and reach the goal (G) for +10 reward."
        )

    def _symbol_meaning(self, ch: str) -> str:
        meanings = {
            ".": "empty",
            "#": "wall",
            "G": "goal",
            "r": "red key",
            "b": "blue key",
            "y": "yellow key",
            "R": "red door (locked)",
            "B": "blue door (locked)",
            "Y": "yellow door (locked)",
        }
        return meanings.get(ch, super()._symbol_meaning(ch))
