"""MiniGrid Dynamic-Obstacles environments.

Empty room with moving obstacle balls.  Agent must reach goal while avoiding
obstacles that bounce around the room.
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.observation import GridObservation
from atlas_rl.envs.minigrid.base import DIR_RIGHT, DIR_TO_VEC, MiniGridBase
from atlas_rl.envs.minigrid.objects import Ball, Goal


class _DynamicObstaclesBase(MiniGridBase):
    _room_size: int = 5  # interior size
    _num_obstacles: int = 2

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._obstacle_positions: list[tuple[int, int]] = []
        self._obstacle_directions: list[int] = []  # direction index 0-3

    # ------------------------------------------------------------------
    # Grid generation
    # ------------------------------------------------------------------

    def _generate_grid(self, seed: int) -> None:
        size = self._room_size + 2
        self._init_grid(size, size)

        # Agent at top-left interior corner
        self._place_agent(1, 1, DIR_RIGHT)

        # Goal at bottom-right interior corner
        self._place_obj(size - 2, size - 2, Goal())

        # Place obstacles at random interior positions
        self._obstacle_positions = []
        self._obstacle_directions = []
        occupied: set[tuple[int, int]] = {(1, 1), (size - 2, size - 2)}

        for _ in range(self._num_obstacles):
            while True:
                ox = int(self.rng.integers(1, size - 1))
                oy = int(self.rng.integers(1, size - 1))
                if (ox, oy) not in occupied:
                    break
            occupied.add((ox, oy))
            self._obstacle_positions.append((ox, oy))
            self._obstacle_directions.append(int(self.rng.integers(0, 4)))
            self._place_obj(ox, oy, Ball(color="blue"))

    # ------------------------------------------------------------------
    # Step: base action then move obstacles
    # ------------------------------------------------------------------

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)

        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        # Move each obstacle one cell
        size = self._room_size + 2
        for i in range(len(self._obstacle_positions)):
            ox, oy = self._obstacle_positions[i]
            d = self._obstacle_directions[i]
            dx, dy = DIR_TO_VEC[d]
            nx, ny = ox + dx, oy + dy

            if not self._can_obstacle_enter(nx, ny, size, i):
                # Reverse direction and try again
                d = (d + 2) % 4
                self._obstacle_directions[i] = d
                dx, dy = DIR_TO_VEC[d]
                nx, ny = ox + dx, oy + dy
                if not self._can_obstacle_enter(nx, ny, size, i):
                    continue  # still blocked -- stay in place

            # Clear old cell (restore goal if obstacle was on goal)
            if (ox, oy) == self._goal_pos:
                self._grid[oy][ox] = Goal()
            else:
                self._grid[oy][ox] = None

            self._obstacle_positions[i] = (nx, ny)

            # Place obstacle in new cell (may overlap goal visually)
            self._grid[ny][nx] = Ball(color="blue")

        # Collision: an obstacle moved onto the agent
        ax, ay = self._agent_pos
        for ox, oy in self._obstacle_positions:
            if (ax, ay) == (ox, oy):
                info["obstacle_collision"] = True
                return self._render_current_observation(), 0.0, True, False, info

        return self._render_current_observation(), reward, terminated, truncated, info

    def _can_obstacle_enter(
        self, nx: int, ny: int, size: int, obstacle_idx: int
    ) -> bool:
        """Return True if obstacle *obstacle_idx* can move to (nx, ny)."""
        # Out of interior bounds (walls)
        if nx <= 0 or nx >= size - 1 or ny <= 0 or ny >= size - 1:
            return False
        # Another obstacle already there
        for j, (px, py) in enumerate(self._obstacle_positions):
            if j != obstacle_idx and (px, py) == (nx, ny):
                return False
        # Solid object that isn't a goal (goals are fine to overlap)
        obj = self._get_obj(nx, ny)
        return obj is None or obj.can_overlap or isinstance(obj, Ball)

    # ------------------------------------------------------------------
    # Task description
    # ------------------------------------------------------------------

    def _task_description(self) -> str:
        return (
            f"Navigate a {self._room_size}x{self._room_size} room to the goal (G) "
            f"while avoiding {self._num_obstacles} moving blue obstacles (O). "
            "Obstacles move one cell per turn and bounce off walls. "
            "Touching an obstacle ends the episode with zero reward. "
            "Reward = 1 - 0.9 * (steps / max_steps)."
        )


# ------------------------------------------------------------------
# Concrete variants
# ------------------------------------------------------------------


class MiniGridDynamicObstacles5x5Env(_DynamicObstaclesBase):
    _room_size = 5
    _num_obstacles = 2

    def env_id(self) -> str:
        return "atlas_rl/minigrid-dynamic-obstacles-5x5-v0"


class MiniGridDynamicObstacles6x6Env(_DynamicObstaclesBase):
    _room_size = 6
    _num_obstacles = 3

    def env_id(self) -> str:
        return "atlas_rl/minigrid-dynamic-obstacles-6x6-v0"


class MiniGridDynamicObstacles8x8Env(_DynamicObstaclesBase):
    _room_size = 8
    _num_obstacles = 4

    def env_id(self) -> str:
        return "atlas_rl/minigrid-dynamic-obstacles-8x8-v0"


class MiniGridDynamicObstacles16x16Env(_DynamicObstaclesBase):
    _room_size = 16
    _num_obstacles = 8

    def env_id(self) -> str:
        return "atlas_rl/minigrid-dynamic-obstacles-16x16-v0"
