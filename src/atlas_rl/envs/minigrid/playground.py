"""MiniGrid Playground environment.

Large room with many diverse objects for free exploration.
"""

from __future__ import annotations

from atlas_rl.envs.minigrid.base import MiniGridBase
from atlas_rl.envs.minigrid.objects import Ball, Box, Door, Goal, Key, Wall

_COLORS = ["red", "green", "blue", "yellow", "purple", "grey"]


class MiniGridPlaygroundEnv(MiniGridBase):
    """Large room with diverse objects scattered around."""

    def env_id(self) -> str:
        return "atlas_rl/minigrid-playground-v0"

    def _generate_grid(self, seed: int) -> None:
        self._init_grid(18, 18)

        occupied: set[tuple[int, int]] = set()

        def _random_free_pos() -> tuple[int, int]:
            while True:
                x = int(self.rng.integers(1, 17))
                y = int(self.rng.integers(1, 17))
                if (x, y) not in occupied and self._get_obj(x, y) is None:
                    occupied.add((x, y))
                    return (x, y)

        # Place various objects
        # 3 keys of different colors
        for i in range(3):
            x, y = _random_free_pos()
            self._place_obj(x, y, Key(color=_COLORS[i]))

        # 3 balls of different colors
        for i in range(3):
            x, y = _random_free_pos()
            self._place_obj(x, y, Ball(color=_COLORS[i + 3]))

        # 2 boxes
        for i in range(2):
            x, y = _random_free_pos()
            self._place_obj(x, y, Box(color=_COLORS[i]))

        # A few internal wall segments for variety
        for _ in range(3):
            wx = int(self.rng.integers(3, 15))
            wy = int(self.rng.integers(3, 15))
            length = int(self.rng.integers(2, 5))
            horizontal = bool(self.rng.integers(0, 2))
            for j in range(length):
                if horizontal:
                    px, py = wx + j, wy
                else:
                    px, py = wx, wy + j
                if (
                    1 <= px < 17
                    and 1 <= py < 17
                    and (px, py) not in occupied
                    and self._get_obj(px, py) is None
                ):
                    self._place_obj(px, py, Wall())
                    occupied.add((px, py))

        # 2 doors (unlocked) on internal walls
        for i in range(2):
            x, y = _random_free_pos()
            self._place_obj(x, y, Door(color=_COLORS[i]))

        # Goal
        gx, gy = _random_free_pos()
        self._place_obj(gx, gy, Goal())

        # Agent
        ax, ay = _random_free_pos()
        direction = int(self.rng.integers(0, 4))
        self._place_agent(ax, ay, direction)

    def _task_description(self) -> str:
        return (
            "A large 16x16 room filled with various objects: keys (K), balls (O), "
            "boxes (B), doors (D/d), and walls (#). Explore freely and find the "
            "goal (G). You can pick up keys and balls, open doors, and drop items. "
            "Reward = 1 - 0.9 * (steps / max_steps)."
        )
