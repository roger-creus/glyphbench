"""MiniGrid RedBlueDoors environments.

Two rooms separated by a wall with a red door and a blue door.
Agent must navigate through the correct door to reach the goal.
"""

from __future__ import annotations

from atlas_rl.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from atlas_rl.envs.minigrid.objects import Door, Goal, Wall


class _RedBlueDoorsBase(MiniGridBase):
    _room_size: int = 6  # interior size per room

    def _generate_grid(self, seed: int) -> None:
        rs = self._room_size
        grid_w = 2 * rs + 3  # two rooms + shared wall + outer walls
        grid_h = rs + 2
        self._init_grid(grid_w, grid_h)

        # Vertical dividing wall
        wall_x = rs + 1
        for y in range(1, grid_h - 1):
            self._place_obj(wall_x, y, Wall())

        # Two distinct door positions in the wall
        pos1 = int(self.rng.integers(1, grid_h - 1))
        pos2 = pos1
        while pos2 == pos1:
            pos2 = int(self.rng.integers(1, grid_h - 1))

        # Place red and blue doors (randomly assign which slot gets which color)
        if self.rng.integers(0, 2) == 0:
            red_y, blue_y = pos1, pos2
        else:
            red_y, blue_y = pos2, pos1

        self._grid[red_y][wall_x] = None
        self._place_obj(wall_x, red_y, Door(color="red"))
        self._grid[blue_y][wall_x] = None
        self._place_obj(wall_x, blue_y, Door(color="blue"))

        # Agent in left room
        ax = int(self.rng.integers(1, wall_x))
        ay = int(self.rng.integers(1, grid_h - 1))
        self._place_agent(ax, ay, DIR_RIGHT)

        # Goal in right room
        gx = int(self.rng.integers(wall_x + 1, grid_w - 1))
        gy = int(self.rng.integers(1, grid_h - 1))
        self._place_obj(gx, gy, Goal())

    def _task_description(self) -> str:
        return (
            "Two rooms separated by a wall with a red door "
            "and a blue door. "
            "Toggle a door open and pass through to reach "
            "the goal in the other room. "
            "Reward = 1 - 0.9 * (steps / max_steps)."
        )


class MiniGridRedBlueDoors6x6Env(_RedBlueDoorsBase):
    _room_size = 6

    def env_id(self) -> str:
        return "atlas_rl/minigrid-redbluedoors-6x6-v0"


class MiniGridRedBlueDoors8x8Env(_RedBlueDoorsBase):
    _room_size = 8

    def env_id(self) -> str:
        return "atlas_rl/minigrid-redbluedoors-8x8-v0"
