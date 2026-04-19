"""MiniGrid Fetch and PutNear environments.

Fetch: pick up a specific target object.
PutNear: pick up an object and drop it near a target position.
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.observation import GridObservation
from atlas_rl.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from atlas_rl.envs.minigrid.objects import Ball, Key

_COLORS = ["red", "green", "blue", "yellow", "purple"]
_OBJ_TYPES: list[type[Ball | Key]] = [Ball, Key]


class _FetchBase(MiniGridBase):
    _room_size: int = 5
    _num_objects: int = 2

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._target_obj: Ball | Key | None = None
        self._target_desc: str = ""

    def _generate_grid(self, seed: int) -> None:
        size = self._room_size + 2
        self._init_grid(size, size)

        occupied: set[tuple[int, int]] = set()
        objects: list[tuple[int, int, Ball | Key]] = []

        for i in range(self._num_objects):
            while True:
                ox = int(self.rng.integers(1, size - 1))
                oy = int(self.rng.integers(1, size - 1))
                if (ox, oy) not in occupied:
                    break
            occupied.add((ox, oy))
            color = _COLORS[i % len(_COLORS)]
            obj_cls = _OBJ_TYPES[i % len(_OBJ_TYPES)]
            obj = obj_cls(color=color)
            self._place_obj(ox, oy, obj)
            objects.append((ox, oy, obj))

        # Pick target
        target_idx = int(self.rng.integers(0, len(objects)))
        _, _, self._target_obj = objects[target_idx]
        self._target_desc = self._target_obj.legend_name()

        # Agent
        while True:
            ax = int(self.rng.integers(1, size - 1))
            ay = int(self.rng.integers(1, size - 1))
            if (ax, ay) not in occupied:
                break
        self._place_agent(ax, ay, DIR_RIGHT)

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)

        # Check if agent just picked up the target object
        if self._carrying is not None and self._carrying is self._target_obj:
            terminated = True
            reward = self._reward_on_goal()
            info["target_fetched"] = True

        return obs, reward, terminated, truncated, info

    def _task_description(self) -> str:
        return (
            f"A room with {self._num_objects} colored objects. Find and pick up "
            f"the {self._target_desc} using PICKUP. "
            f"Reward = 1 - 0.9 * (steps / max_steps)."
        )


class MiniGridFetch5x5N2Env(_FetchBase):
    _room_size = 5
    _num_objects = 2

    def env_id(self) -> str:
        return "atlas_rl/minigrid-fetch-5x5-n2-v0"


class MiniGridFetch6x6N2Env(_FetchBase):
    _room_size = 6
    _num_objects = 2

    def env_id(self) -> str:
        return "atlas_rl/minigrid-fetch-6x6-n2-v0"


class MiniGridFetch8x8N3Env(_FetchBase):
    _room_size = 8
    _num_objects = 3

    def env_id(self) -> str:
        return "atlas_rl/minigrid-fetch-8x8-n3-v0"


class _PutNearBase(MiniGridBase):
    _room_size: int = 6
    _num_objects: int = 2

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._target_obj: Ball | Key | None = None
        self._target_desc: str = ""
        self._goal_pos_putnear: tuple[int, int] = (0, 0)
        self._goal_desc: str = ""

    def _generate_grid(self, seed: int) -> None:
        size = self._room_size + 2
        self._init_grid(size, size)

        occupied: set[tuple[int, int]] = set()
        objects: list[tuple[int, int, Ball | Key]] = []

        for i in range(self._num_objects):
            while True:
                ox = int(self.rng.integers(1, size - 1))
                oy = int(self.rng.integers(1, size - 1))
                if (ox, oy) not in occupied:
                    break
            occupied.add((ox, oy))
            color = _COLORS[i % len(_COLORS)]
            obj_cls = _OBJ_TYPES[i % len(_OBJ_TYPES)]
            obj = obj_cls(color=color)
            self._place_obj(ox, oy, obj)
            objects.append((ox, oy, obj))

        # Pick target to carry
        target_idx = int(self.rng.integers(0, len(objects)))
        _, _, self._target_obj = objects[target_idx]
        self._target_desc = self._target_obj.legend_name()

        # Pick goal position (where to put near) — the other object
        goal_idx = (target_idx + 1) % len(objects)
        gx, gy, goal_obj = objects[goal_idx]
        self._goal_pos_putnear = (gx, gy)
        self._goal_desc = goal_obj.legend_name()

        # Agent
        while True:
            ax = int(self.rng.integers(1, size - 1))
            ay = int(self.rng.integers(1, size - 1))
            if (ax, ay) not in occupied:
                break
        self._place_agent(ax, ay, DIR_RIGHT)

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        was_carrying = self._carrying
        obs, reward, terminated, truncated, info = super()._step(action)

        # Check if agent just dropped the target object near the goal
        if was_carrying is self._target_obj and self._carrying is None:
            # Agent dropped something. Check if it landed adjacent to goal position.
            gx, gy = self._goal_pos_putnear
            fx, fy = self._front_pos()
            if abs(fx - gx) <= 1 and abs(fy - gy) <= 1:
                terminated = True
                reward = self._reward_on_goal()
                info["put_near_success"] = True

        return obs, reward, terminated, truncated, info

    def _task_description(self) -> str:
        return (
            f"A room with {self._num_objects} objects. Pick up the {self._target_desc} "
            f"and drop it near the {self._goal_desc}. "
            f"Reward = 1 - 0.9 * (steps / max_steps)."
        )


class MiniGridPutNear6x6N2Env(_PutNearBase):
    _room_size = 6
    _num_objects = 2

    def env_id(self) -> str:
        return "atlas_rl/minigrid-putnear-6x6-n2-v0"


class MiniGridPutNear8x8N3Env(_PutNearBase):
    _room_size = 8
    _num_objects = 3

    def env_id(self) -> str:
        return "atlas_rl/minigrid-putnear-8x8-n3-v0"
