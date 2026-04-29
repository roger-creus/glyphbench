"""MiniGrid Crossing and SimpleCrossing environments.

Crossing: horizontal lava/water strips with gaps.
SimpleCrossing: horizontal wall strips with gaps.
"""

from __future__ import annotations

from glyphbench.envs.minigrid.base import DIR_UP, MiniGridBase
from glyphbench.envs.minigrid.objects import Goal, Lava, Wall, Water


class _CrossingBase(MiniGridBase):
    _num_strips: int = 1
    _obstacle_type: str = "lava"  # "lava", "water", or "wall"
    _easy: bool = False  # if True, gaps always at center

    def _generate_grid(self, seed: int) -> None:
        # Scale grid so strips never overlap the goal row (y=1)
        size = max(9, 2 * self._num_strips + 5)
        self._init_grid(size, size)

        # Place obstacle strips at evenly spaced y positions
        strip_spacing = (size - 2) // (self._num_strips + 1)
        for i in range(self._num_strips):
            strip_y = strip_spacing * (i + 1)
            # Fill strip
            for x in range(1, size - 1):
                if self._obstacle_type == "lava":
                    self._place_obj(x, strip_y, Lava())
                elif self._obstacle_type == "water":
                    self._place_obj(x, strip_y, Water())
                else:  # wall
                    self._place_obj(x, strip_y, Wall())

            # Create gap
            gap_x = size // 2 if self._easy else int(self.rng.integers(1, size - 1))
            self._grid[strip_y][gap_x] = None  # remove obstacle at gap

        # Agent at bottom center
        self._place_agent(size // 2, size - 2, DIR_UP)

        # Goal at top center
        self._place_obj(size // 2, 1, Goal())

    def _task_description(self) -> str:
        obs_name = {
            "lava": f"lava ({Lava().render_char()})",
            "water": f"water ({Water().render_char()})",
            "wall": f"walls ({Wall().render_char()})",
        }.get(self._obstacle_type, self._obstacle_type)
        danger = ""
        if self._obstacle_type == "lava":
            danger = " Stepping on lava ends the episode with zero reward."
        goal = Goal().render_char()
        return (
            f"Navigate through {self._num_strips} horizontal strip(s) of "
            f"{obs_name} to reach the goal ({goal}). Each strip has one gap "
            f"you can pass through.{danger} "
            f"Reward = 1 - 0.9 * (steps / max_steps)."
        )


# Lava Crossing variants
class MiniGridCrossingN1Env(_CrossingBase):
    _num_strips = 1
    _obstacle_type = "lava"

    def env_id(self) -> str:
        return "glyphbench/minigrid-crossing-n1-v0"


class MiniGridCrossingN2Env(_CrossingBase):
    _num_strips = 2
    _obstacle_type = "lava"

    def env_id(self) -> str:
        return "glyphbench/minigrid-crossing-n2-v0"


class MiniGridCrossingN3Env(_CrossingBase):
    _num_strips = 3
    _obstacle_type = "lava"

    def env_id(self) -> str:
        return "glyphbench/minigrid-crossing-n3-v0"


# Safe (water) Crossing variants
class MiniGridCrossingN1SafeEnv(_CrossingBase):
    _num_strips = 1
    _obstacle_type = "water"

    def env_id(self) -> str:
        return "glyphbench/minigrid-crossing-n1-safe-v0"


class MiniGridCrossingN2SafeEnv(_CrossingBase):
    _num_strips = 2
    _obstacle_type = "water"

    def env_id(self) -> str:
        return "glyphbench/minigrid-crossing-n2-safe-v0"


class MiniGridCrossingN3SafeEnv(_CrossingBase):
    _num_strips = 3
    _obstacle_type = "water"

    def env_id(self) -> str:
        return "glyphbench/minigrid-crossing-n3-safe-v0"


# SimpleCrossing (wall) variants
class MiniGridSimpleCrossingN1Env(_CrossingBase):
    _num_strips = 1
    _obstacle_type = "wall"

    def env_id(self) -> str:
        return "glyphbench/minigrid-simplecrossing-n1-v0"


class MiniGridSimpleCrossingN2Env(_CrossingBase):
    _num_strips = 2
    _obstacle_type = "wall"

    def env_id(self) -> str:
        return "glyphbench/minigrid-simplecrossing-n2-v0"


class MiniGridSimpleCrossingN3Env(_CrossingBase):
    _num_strips = 3
    _obstacle_type = "wall"

    def env_id(self) -> str:
        return "glyphbench/minigrid-simplecrossing-n3-v0"


# SimpleCrossing Easy variants (gaps always at center)
class MiniGridSimpleCrossingEasyN1Env(_CrossingBase):
    _num_strips = 1
    _obstacle_type = "wall"
    _easy = True

    def env_id(self) -> str:
        return "glyphbench/minigrid-simplecrossing-easy-n1-v0"


class MiniGridSimpleCrossingEasyN2Env(_CrossingBase):
    _num_strips = 2
    _obstacle_type = "wall"
    _easy = True

    def env_id(self) -> str:
        return "glyphbench/minigrid-simplecrossing-easy-n2-v0"


class MiniGridSimpleCrossingEasyN3Env(_CrossingBase):
    _num_strips = 3
    _obstacle_type = "wall"
    _easy = True

    def env_id(self) -> str:
        return "glyphbench/minigrid-simplecrossing-easy-n3-v0"
