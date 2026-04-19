"""MiniHack Room environments.

Room variants of increasing difficulty, all built on MiniHackBase.

Gym IDs: atlas_rl/minihack-room-{variant}-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.observation import GridObservation
from atlas_rl.envs.minihack.base import MiniHackBase
from atlas_rl.envs.minihack.creatures import KOBOLD, ORC, RAT


class _RoomBase(MiniHackBase):
    """Shared logic for all Room variants."""

    _room_size: int = 5  # interior size
    _has_monsters: bool = False
    _has_traps: bool = False
    _is_dark: bool = False
    _random_start: bool = True  # original Room-5x5 had random start

    def _generate_level(self, seed: int) -> None:
        size = self._room_size + 2
        self._init_grid(size, size)
        self._dark = self._is_dark

        interior = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
        ]
        occupied: set[tuple[int, int]] = set()

        # Player position
        if self._random_start:
            idx = int(self.rng.integers(0, len(interior)))
            px, py = interior[idx]
        else:
            px, py = 1, 1
        self._place_player(px, py)
        occupied.add((px, py))

        # Goal (stairs) at random position, not on player
        while True:
            idx = int(self.rng.integers(0, len(interior)))
            gx, gy = interior[idx]
            if (gx, gy) not in occupied:
                break
        self._place_stairs(gx, gy)
        occupied.add((gx, gy))

        # Traps
        if self._has_traps:
            n_traps = (
                int(self.rng.integers(2, 4))  # 2-3
                if self._room_size <= 5
                else int(self.rng.integers(5, 9))  # 5-8
            )
            for _ in range(n_traps):
                while True:
                    idx = int(self.rng.integers(0, len(interior)))
                    tx, ty = interior[idx]
                    if (tx, ty) not in occupied:
                        break
                self._place_trap(tx, ty)
                occupied.add((tx, ty))

        # Monsters
        if self._has_monsters:
            n_monsters = (
                int(self.rng.integers(1, 3))  # 1-2
                if self._room_size <= 5
                else int(self.rng.integers(3, 6))  # 3-5
            )
            monster_types = [RAT, KOBOLD, ORC]
            for i in range(n_monsters):
                while True:
                    idx = int(self.rng.integers(0, len(interior)))
                    mx, my = interior[idx]
                    if (mx, my) not in occupied:
                        break
                ctype = monster_types[i % len(monster_types)]
                self._spawn_creature(ctype, mx, my)
                occupied.add((mx, my))

    def _task_description(self) -> str:
        parts = [
            f"Navigate a {self._room_size}x{self._room_size} "
            f"dungeon room to the stairs (>)."
        ]
        if self._has_monsters:
            parts.append(
                "Hostile monsters roam the room -- fight or avoid them."
            )
        if self._has_traps:
            parts.append(
                "Traps (^) are scattered around -- stepping on one deals damage."
            )
        if self._is_dark:
            parts.append(
                "The room is dark -- you can only see tiles adjacent to you."
            )
        parts.append("Reward: +1 on reaching stairs, -1 on death.")
        return " ".join(parts)


# ------------------------------------------------------------------
# Backward-compat wrapper so existing Room-5x5 tests keep working.
# The old MiniHackRoom5x5Env exposed _agent_x/y, _goal_x/y and
# returned room_size / goal_pos in info.  We patch _step to include
# those extras, and expose the old attributes as properties.
# ------------------------------------------------------------------


class MiniHackRoom5x5Env(_RoomBase):
    """MiniHack Room-5x5: 5x5 interior (7x7 grid)."""

    _room_size = 5
    _random_start = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-room-5x5-v0"

    # Backward-compat properties for tests that poke internals
    @property
    def _agent_x(self) -> int:
        return self._player_pos[0]

    @property
    def _agent_y(self) -> int:
        return self._player_pos[1]

    @property
    def _goal_x(self) -> int:
        return self._goal_pos[0] if self._goal_pos else 0

    @property
    def _goal_y(self) -> int:
        return self._goal_pos[1] if self._goal_pos else 0

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        # Original env never had negative rewards (no monsters/traps)
        info["agent_pos"] = (
            self._player_pos[0] - 1,
            self._player_pos[1] - 1,
        )
        info["goal_pos"] = (
            (self._goal_pos[0] - 1, self._goal_pos[1] - 1)
            if self._goal_pos
            else (-1, -1)
        )
        info["room_size"] = (5, 5)
        info["steps_to_goal"] = (self._turn + 1) if info.get("goal_reached") else -1
        return obs, reward, terminated, truncated, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        # Patch HUD to match old format (includes AC and $)
        px, py = self._player_pos
        hud = (
            f"Dlvl: 1    HP: {self._player_hp}/{self._player_max_hp}    AC: 10    "
            f"Turn: {self._turn}    $: 0    "
            f"Pos: ({px - 1},{py - 1})"
        )
        return GridObservation(
            grid=obs.grid,
            legend=obs.legend,
            hud=hud,
            message=obs.message,
        )


# ------------------------------------------------------------------
# Room-15x15
# ------------------------------------------------------------------


class MiniHackRoom15x15Env(_RoomBase):
    """MiniHack Room-15x15: 15x15 interior (17x17 grid)."""

    _room_size = 15
    _random_start = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-room-15x15-v0"


# ------------------------------------------------------------------
# Room-Random variants (explicit random start -- same as base, kept
# for parity with MiniHack naming convention)
# ------------------------------------------------------------------


class MiniHackRoomRandom5x5Env(_RoomBase):
    """MiniHack Room-Random-5x5: random player start in a 5x5 room."""

    _room_size = 5
    _random_start = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-room-random-5x5-v0"


class MiniHackRoomRandom15x15Env(_RoomBase):
    """MiniHack Room-Random-15x15: random player start in a 15x15 room."""

    _room_size = 15
    _random_start = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-room-random-15x15-v0"


# ------------------------------------------------------------------
# Room-Dark variants
# ------------------------------------------------------------------


class MiniHackRoomDark5x5Env(_RoomBase):
    """MiniHack Room-Dark-5x5: dark 5x5 room with limited vision."""

    _room_size = 5
    _random_start = True
    _is_dark = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-room-dark-5x5-v0"


class MiniHackRoomDark15x15Env(_RoomBase):
    """MiniHack Room-Dark-15x15: dark 15x15 room with limited vision."""

    _room_size = 15
    _random_start = True
    _is_dark = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-room-dark-15x15-v0"


# ------------------------------------------------------------------
# Room-Monster variants
# ------------------------------------------------------------------


class MiniHackRoomMonster5x5Env(_RoomBase):
    """MiniHack Room-Monster-5x5: 5x5 room with 1-2 hostile monsters."""

    _room_size = 5
    _random_start = True
    _has_monsters = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-room-monster-5x5-v0"


class MiniHackRoomMonster15x15Env(_RoomBase):
    """MiniHack Room-Monster-15x15: 15x15 room with 3-5 hostile monsters."""

    _room_size = 15
    _random_start = True
    _has_monsters = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-room-monster-15x15-v0"


# ------------------------------------------------------------------
# Room-Trap variants
# ------------------------------------------------------------------


class MiniHackRoomTrap5x5Env(_RoomBase):
    """MiniHack Room-Trap-5x5: 5x5 room with 2-3 traps."""

    _room_size = 5
    _random_start = True
    _has_traps = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-room-trap-5x5-v0"


class MiniHackRoomTrap15x15Env(_RoomBase):
    """MiniHack Room-Trap-15x15: 15x15 room with 5-8 traps."""

    _room_size = 15
    _random_start = True
    _has_traps = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-room-trap-15x15-v0"


# ------------------------------------------------------------------
# Room-Ultimate variants (monsters + traps + dark)
# ------------------------------------------------------------------


class MiniHackRoomUltimate5x5Env(_RoomBase):
    """MiniHack Room-Ultimate-5x5: dark 5x5 room with monsters and traps."""

    _room_size = 5
    _random_start = True
    _has_monsters = True
    _has_traps = True
    _is_dark = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-room-ultimate-5x5-v0"


class MiniHackRoomUltimate15x15Env(_RoomBase):
    """MiniHack Room-Ultimate-15x15: dark 15x15 room with monsters and traps."""

    _room_size = 15
    _random_start = True
    _has_monsters = True
    _has_traps = True
    _is_dark = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-room-ultimate-15x15-v0"
