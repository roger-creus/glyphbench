"""MiniHack Room-5x5 environment.

A 5x5 dungeon room (7x7 grid including walls). Random start and goal
positions (seeded). Agent must reach the stairs down (>).

Gym ID: atlas_rl/minihack-room-5x5-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.ascii_primitives import (
    build_legend,
    grid_to_string,
    make_empty_grid,
    stamp_sprite,
)
from atlas_rl.core.base_env import BaseAsciiEnv
from atlas_rl.core.observation import GridObservation
from atlas_rl.envs.minihack.base import (
    MINIHACK_ACTION_SPEC,
    MOVE_VECTORS,
)


class MiniHackRoom5x5Env(BaseAsciiEnv):
    """MiniHack Room-5x5: navigate a 5x5 dungeon room to reach the stairs.

    Grid is 7x7 (5x5 interior + wall border). Agent (@) and goal (>) are
    placed randomly within the interior on each reset (seeded).

    Actions: 8 directional moves + 7 utility actions (mostly no-ops).
    Reward: +1 on reaching the goal, 0 otherwise.
    """

    action_spec = MINIHACK_ACTION_SPEC
    noop_action_name = "WAIT"

    _GRID_W = 7
    _GRID_H = 7
    _INTERIOR_MIN = 1  # inclusive
    _INTERIOR_MAX = 5  # inclusive

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._agent_x: int = 1
        self._agent_y: int = 1
        self._goal_x: int = 5
        self._goal_y: int = 5
        self._message: str = ""

    def env_id(self) -> str:
        return "atlas_rl/minihack-room-5x5-v0"

    def system_prompt(self) -> str:
        return (
            "You are playing MiniHack Room-5x5.\n\n"
            "TASK\n"
            "Navigate a 5x5 dungeon room to reach the stairs down (>). "
            "You earn +1 reward for reaching the stairs. The agent (@) and "
            "stairs are placed randomly each episode.\n\n"
            "GRID\n"
            "The room is a 7x7 grid (5x5 interior + walls). @ is you. > is "
            "the stairs down (goal). . is dungeon floor. | and - are walls.\n\n"
            "MOVEMENT\n"
            "You can move in 8 directions: N, S, E, W, NE, NW, SE, SW. "
            "Moving into a wall does nothing. WAIT, SEARCH, LOOK, PICKUP, "
            "APPLY, INVENTORY, and ESCAPE are all no-ops in this room.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        # Random positions for agent and goal within interior (1-5)
        interior_cells = [
            (x, y)
            for x in range(self._INTERIOR_MIN, self._INTERIOR_MAX + 1)
            for y in range(self._INTERIOR_MIN, self._INTERIOR_MAX + 1)
        ]
        indices = self.rng.choice(len(interior_cells), size=2, replace=False)
        self._agent_x, self._agent_y = interior_cells[int(indices[0])]
        self._goal_x, self._goal_y = interior_cells[int(indices[1])]
        self._message = ""
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        self._message = ""

        if name in MOVE_VECTORS:
            dx, dy = MOVE_VECTORS[name]
            new_x = self._agent_x + dx
            new_y = self._agent_y + dy
            # Bounds check: interior is 1-5
            if (self._INTERIOR_MIN <= new_x <= self._INTERIOR_MAX
                    and self._INTERIOR_MIN <= new_y <= self._INTERIOR_MAX):
                self._agent_x = new_x
                self._agent_y = new_y
        # All other actions (WAIT, SEARCH, LOOK, etc.) are no-ops

        reached_goal = (
            self._agent_x == self._goal_x and self._agent_y == self._goal_y
        )
        reward = 1.0 if reached_goal else 0.0
        terminated = reached_goal

        if reached_goal:
            self._message = "You see a staircase leading down. You descend."

        info: dict[str, Any] = {
            "agent_pos": (self._agent_x - 1, self._agent_y - 1),  # interior coords
            "goal_pos": (self._goal_x - 1, self._goal_y - 1),
            "goal_reached": reached_goal,
            "room_size": (5, 5),
            "steps_to_goal": (self._turn + 1) if reached_goal else -1,
        }

        return self._render_current_observation(), reward, terminated, False, info

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(self._GRID_W, self._GRID_H, fill=".")

        # Draw walls: top and bottom rows are -, left and right cols are |
        for x in range(self._GRID_W):
            grid[0][x] = "-"
            grid[self._GRID_H - 1][x] = "-"
        for y in range(self._GRID_H):
            grid[y][0] = "|"
            grid[y][self._GRID_W - 1] = "|"
        # Corners: keep as - (NetHack convention: - for horizontal walls including corners)
        grid[0][0] = "-"
        grid[0][self._GRID_W - 1] = "-"
        grid[self._GRID_H - 1][0] = "-"
        grid[self._GRID_H - 1][self._GRID_W - 1] = "-"

        stamp_sprite(grid, self._goal_x, self._goal_y, ">")
        stamp_sprite(grid, self._agent_x, self._agent_y, "@")

        hud = (
            f"Dlvl: 1    HP: 12/12    AC: 10    "
            f"Turn: {self._turn}    $: 0    "
            f"Pos: ({self._agent_x - 1},{self._agent_y - 1})"
        )

        legend = build_legend({
            "@": "you",
            ">": "stairs down (goal)",
            ".": "dungeon floor",
            "|": "wall (vertical)",
            "-": "wall (horizontal)",
        })

        return GridObservation(
            grid=grid_to_string(grid),
            legend=legend,
            hud=hud,
            message=self._message,
        )
