"""MiniGrid GoToDoor and GoToObject environments.

GoToDoor: navigate to a specific colored door.
GoToObject: navigate to a specific colored object.
"""

from __future__ import annotations

from atlas_rl.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from atlas_rl.envs.minigrid.objects import Ball, Door, Goal, Key

_COLORS = ["red", "green", "blue", "yellow", "purple"]


class _GoToDoorBase(MiniGridBase):
    _room_size: int = 5  # interior size

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._target_color: str = "red"

    def _generate_grid(self, seed: int) -> None:
        size = self._room_size + 2
        self._init_grid(size, size)

        # Place 4 colored doors on the walls (one per wall)
        colors_used = list(_COLORS[:4])
        self.rng.shuffle(colors_used)

        door_positions = [
            (size // 2, 0),            # top wall
            (size // 2, size - 1),     # bottom wall
            (0, size // 2),            # left wall
            (size - 1, size // 2),     # right wall
        ]

        doors = []
        for i, (dx, dy) in enumerate(door_positions):
            color = colors_used[i]
            door = Door(color=color)
            self._grid[dy][dx] = door  # overwrite wall with door
            doors.append((dx, dy, color))

        # Pick target door
        target_idx = int(self.rng.integers(0, len(doors)))
        self._target_color = doors[target_idx][2]
        tx, ty = doors[target_idx][0], doors[target_idx][1]

        # Place goal adjacent to target door (inside the room)
        for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            gx, gy = tx + ddx, ty + ddy
            if 1 <= gx < size - 1 and 1 <= gy < size - 1:
                self._place_obj(gx, gy, Goal())
                break

        # Agent at random interior position, not on goal
        while True:
            ax = int(self.rng.integers(1, size - 1))
            ay = int(self.rng.integers(1, size - 1))
            if self._get_obj(ax, ay) is None:
                break
        self._place_agent(ax, ay, DIR_RIGHT)

    def _task_description(self) -> str:
        return (
            f"A room with colored doors on each wall. Navigate to the "
            f"{self._target_color} door. "
            f"Reward = 1 - 0.9 * (steps / max_steps)."
        )


class MiniGridGoToDoor5x5Env(_GoToDoorBase):
    _room_size = 5

    def env_id(self) -> str:
        return "atlas_rl/minigrid-gotodoor-5x5-v0"


class MiniGridGoToDoor6x6Env(_GoToDoorBase):
    _room_size = 6

    def env_id(self) -> str:
        return "atlas_rl/minigrid-gotodoor-6x6-v0"


class MiniGridGoToDoor8x8Env(_GoToDoorBase):
    _room_size = 8

    def env_id(self) -> str:
        return "atlas_rl/minigrid-gotodoor-8x8-v0"


class MiniGridGoToObject6x6N2Env(MiniGridBase):
    """Room with N colored objects. Navigate to the target object."""

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._target_desc: str = ""

    def env_id(self) -> str:
        return "atlas_rl/minigrid-gotoobject-6x6-n2-v0"

    def _generate_grid(self, seed: int) -> None:
        self._init_grid(8, 8)

        # Place 2 colored objects (ball and key)
        obj_types = [Ball, Key]
        colors = list(_COLORS[:2])
        self.rng.shuffle(colors)

        objects = []
        occupied: set[tuple[int, int]] = set()
        for i in range(2):
            while True:
                ox = int(self.rng.integers(1, 7))
                oy = int(self.rng.integers(1, 7))
                if (ox, oy) not in occupied:
                    break
            occupied.add((ox, oy))
            obj_class = obj_types[i % 2]
            obj = obj_class(color=colors[i])
            self._place_obj(ox, oy, obj)
            objects.append((ox, oy, obj))

        # Pick target object
        target_idx = int(self.rng.integers(0, 2))
        tx, ty, target_obj = objects[target_idx]
        self._target_desc = f"{target_obj.obj_type} ({target_obj.color})"

        # Place goal adjacent to target object
        for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]:
            gx, gy = tx + ddx, ty + ddy
            if 1 <= gx < 7 and 1 <= gy < 7 and self._get_obj(gx, gy) is None:
                self._place_obj(gx, gy, Goal())
                break

        # Agent at random interior position, not on any placed object
        while True:
            ax = int(self.rng.integers(1, 7))
            ay = int(self.rng.integers(1, 7))
            if (ax, ay) not in occupied and self._get_obj(ax, ay) is None:
                break
        self._place_agent(ax, ay, DIR_RIGHT)

    def _task_description(self) -> str:
        return (
            f"A room with colored objects. Navigate to the {self._target_desc}. "
            f"Reward = 1 - 0.9 * (steps / max_steps)."
        )
