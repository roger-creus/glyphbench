"""MiniGrid Memory environments.

Long corridor with a colored object at one end and the goal at the other.
Tests the agent's ability to remember information over long trajectories.
"""

from __future__ import annotations

from glyphbench.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from glyphbench.envs.minigrid.objects import Ball, Goal, Key, Wall

_COLORS = ["red", "green", "blue", "yellow", "purple"]
_OBJ_TYPES = [Ball, Key]


class _MemoryBase(MiniGridBase):
    _grid_size: int = 7

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._memory_object_desc: str = ""

    def _generate_grid(self, seed: int) -> None:
        s = self._grid_size
        self._init_grid(s, s)

        # Single horizontal corridor running through the middle row
        corridor_y = s // 2

        # Fill interior rows above and below the corridor with walls
        for y in range(1, s - 1):
            if y != corridor_y:
                for x in range(1, s - 1):
                    self._place_obj(x, y, Wall())

        # Pick a random colored object (key or ball) to serve as the memory cue
        color = _COLORS[int(self.rng.integers(0, len(_COLORS)))]
        obj_cls = _OBJ_TYPES[int(self.rng.integers(0, len(_OBJ_TYPES)))]
        memory_obj = obj_cls(color=color)
        self._memory_object_desc = memory_obj.legend_name()

        # Place the memory object just ahead of the agent (x=2) so it is
        # immediately visible, and position the agent at the western end (x=1).
        # The agent faces right along the corridor toward the goal at x=s-2.
        self._place_obj(2, corridor_y, memory_obj)
        self._place_agent(1, corridor_y, DIR_RIGHT)

        # Goal at the far (eastern) end of the corridor
        self._place_obj(s - 2, corridor_y, Goal())

    def _task_description(self) -> str:
        goal = Goal().render_char()
        return (
            f"A long corridor with a {self._memory_object_desc} near the start. "
            f"Navigate east through the corridor to reach the goal ({goal}) at the far "
            f"end. Remember the object you saw — it may be relevant later. "
            f"Reward = 1 - 0.9 * (steps / max_steps)."
        )


class MiniGridMemoryS7Env(_MemoryBase):
    """Memory corridor in a 7x7 grid."""

    _grid_size = 7

    def env_id(self) -> str:
        return "glyphbench/minigrid-memory-s7-v0"


class MiniGridMemoryS9Env(_MemoryBase):
    """Memory corridor in a 9x9 grid."""

    _grid_size = 9

    def env_id(self) -> str:
        return "glyphbench/minigrid-memory-s9-v0"


class MiniGridMemoryS11Env(_MemoryBase):
    """Memory corridor in an 11x11 grid."""

    _grid_size = 11

    def env_id(self) -> str:
        return "glyphbench/minigrid-memory-s11-v0"


class MiniGridMemoryS13Env(_MemoryBase):
    """Memory corridor in a 13x13 grid."""

    _grid_size = 13

    def env_id(self) -> str:
        return "glyphbench/minigrid-memory-s13-v0"


class MiniGridMemoryS17Env(_MemoryBase):
    """Memory corridor in a 17x17 grid."""

    _grid_size = 17

    def env_id(self) -> str:
        return "glyphbench/minigrid-memory-s17-v0"
