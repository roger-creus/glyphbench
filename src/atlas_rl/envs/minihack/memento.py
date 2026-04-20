"""MiniHack Memento environments. Multi-room/multi-floor navigation."""

from __future__ import annotations

from typing import Any

from atlas_rl.core.observation import GridObservation
from atlas_rl.envs.minihack.base import MiniHackBase


class _MementoBase(MiniHackBase):
    """Base for Memento (procedural multi-room/multi-floor) environments.

    The agent navigates through connected rooms to reach stairs.
    Multi-floor variants require descending stairs multiple times.
    """

    _num_rooms: int = 2
    _num_floors: int = 1

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._current_floor: int = 1
        self._floors_completed: int = 0

    def _generate_level(self, seed: int) -> None:
        self._current_floor = 1
        self._floors_completed = 0
        self._generate_floor(seed)

    def _generate_floor(self, seed: int) -> None:
        """Generate a multi-room dungeon for the current floor."""
        room_w = 5
        total_w = (room_w + 1) * self._num_rooms + 1
        total_h = room_w + 2
        self._init_grid(total_w, total_h)

        # Fill interior with walls, then carve rooms
        for y in range(1, total_h - 1):
            for x in range(1, total_w - 1):
                self._place_wall(x, y)

        # Carve rooms
        for i in range(self._num_rooms):
            rx = (room_w + 1) * i + 1
            for y in range(1, room_w + 1):
                for x in range(rx, rx + room_w):
                    if x < total_w - 1:
                        self._grid[y][x] = "."

        # Corridors between rooms
        cy = total_h // 2
        for i in range(self._num_rooms - 1):
            cx = (room_w + 1) * (i + 1)
            self._grid[cy][cx] = "."

        # Player in first room
        self._place_player(2, total_h // 2)

        # Stairs in last room
        last_x = (room_w + 1) * (self._num_rooms - 1) + room_w
        self._place_stairs(min(last_x, total_w - 2), total_h // 2)

    # ------------------------------------------------------------------
    # Multi-floor step override
    # ------------------------------------------------------------------

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)

        if terminated and info.get("goal_reached"):
            self._floors_completed += 1
            if self._floors_completed < self._num_floors:
                # Generate next floor
                terminated = False
                reward = 0.0
                self._current_floor += 1
                floor_seed = int(self.rng.integers(0, 2**31))
                self._creatures = []
                self._generate_floor(floor_seed)
                self._message = f"You descend to floor {self._current_floor}."
                info["goal_reached"] = False
                info["floor"] = self._current_floor
                obs = self._render_current_observation()
            else:
                reward = 1.0
                self._message = "You have completed all floors!"

        info["floor"] = self._current_floor
        info["floors_completed"] = self._floors_completed
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation: show current floor in HUD
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        floor_info = (
            f"Floor: {self._current_floor}/{self._num_floors}"
        )
        new_hud = obs.hud + "    " + floor_info
        return GridObservation(
            grid=obs.grid,
            legend=obs.legend,
            hud=new_hud,
            message=obs.message,
        )

    # ------------------------------------------------------------------
    # Task description
    # ------------------------------------------------------------------

    def _task_description(self) -> str:
        if self._num_floors > 1:
            return (
                f"Navigate through {self._num_rooms} rooms to reach the stairs "
                f"on each floor. You must descend {self._num_floors} floors "
                f"total. Reward: +1 after completing all floors."
            )
        return (
            f"Navigate through {self._num_rooms} connected rooms to reach "
            f"the stairs (>). Reward: +1 on reaching stairs."
        )


class MiniHackMementoShortEnv(_MementoBase):
    """2 rooms, 1 floor."""

    _num_rooms = 2
    _num_floors = 1

    def env_id(self) -> str:
        return "atlas_rl/minihack-memento-short-v0"


class MiniHackMementoHardEnv(_MementoBase):
    """4 rooms, 1 floor."""

    _num_rooms = 4
    _num_floors = 1

    def env_id(self) -> str:
        return "atlas_rl/minihack-memento-hard-v0"


class MiniHackMementoF2Env(_MementoBase):
    """3 rooms, 2 floors."""

    _num_rooms = 3
    _num_floors = 2

    def env_id(self) -> str:
        return "atlas_rl/minihack-memento-f2-v0"


class MiniHackMementoF4Env(_MementoBase):
    """3 rooms, 4 floors."""

    _num_rooms = 3
    _num_floors = 4

    def env_id(self) -> str:
        return "atlas_rl/minihack-memento-f4-v0"
