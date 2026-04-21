"""MiniGrid DistShift environments.

A strip of lava cuts across the grid. Agent must find the gap to cross safely.
DistShift1 and DistShift2 have the lava at different positions.
"""

from __future__ import annotations

from glyphbench.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from glyphbench.envs.minigrid.objects import Goal, Lava


class _DistShiftBase(MiniGridBase):
    _lava_x: int = 4  # x position of lava strip

    def _generate_grid(self, seed: int) -> None:
        self._init_grid(9, 7)

        # Lava strip at _lava_x
        for y in range(1, 6):
            self._place_obj(self._lava_x, y, Lava())

        # Gap in the lava
        gap_y = int(self.rng.integers(1, 6))
        self._grid[gap_y][self._lava_x] = None  # remove lava to create gap

        # Agent on the left side
        ax = int(self.rng.integers(1, self._lava_x))
        ay = int(self.rng.integers(1, 6))
        self._place_agent(ax, ay, DIR_RIGHT)

        # Goal on the right side
        gx = int(self.rng.integers(self._lava_x + 1, 8))
        gy = int(self.rng.integers(1, 6))
        self._place_obj(gx, gy, Goal())

    def _task_description(self) -> str:
        lava = Lava().render_char()
        goal = Goal().render_char()
        return (
            f"A strip of lava ({lava}) cuts across the grid. Find the gap in the lava "
            f"and cross to reach the goal ({goal}). Stepping on lava ends the episode "
            "with zero reward. Reward = 1 - 0.9 * (steps / max_steps) on reaching goal."
        )


class MiniGridDistShift1Env(_DistShiftBase):
    _lava_x = 4

    def env_id(self) -> str:
        return "glyphbench/minigrid-distshift1-v0"


class MiniGridDistShift2Env(_DistShiftBase):
    _lava_x = 5

    def env_id(self) -> str:
        return "glyphbench/minigrid-distshift2-v0"
