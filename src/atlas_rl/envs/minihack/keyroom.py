"""MiniHack KeyRoom environments.

A room with a locked door (+) blocking the path to the stairs.
The agent must move into the door to open it (changes + to .),
then proceed to the stairs behind it.

Variants:
  - S5: 5x5 interior room
  - S15: 15x15 interior room
  - Dark-S5: S5 with limited visibility
  - Dark-S15: S15 with limited visibility

Gym IDs:
  atlas_rl/minihack-keyroom-s5-v0
  atlas_rl/minihack-keyroom-s15-v0
  atlas_rl/minihack-keyroom-dark-s5-v0
  atlas_rl/minihack-keyroom-dark-s15-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.observation import GridObservation
from atlas_rl.envs.minihack.base import MOVE_VECTORS, MiniHackBase


class _KeyRoomBase(MiniHackBase):
    """Base class for KeyRoom environments.

    Layout: a room with a vertical wall partition in the middle.
    A door (+) in the partition. Player on the left side, stairs on the right.
    Moving into the door opens it (replaces + with .).
    """

    _room_interior: int = 5
    _is_dark: bool = False

    def _generate_level(self, seed: int) -> None:
        ri = self._room_interior
        w = ri + 2  # total grid width including border walls
        h = ri + 2

        self._init_grid(w, h)
        self._dark = self._is_dark

        # Place a vertical partition wall in the middle of the room
        partition_x = w // 2
        for y in range(1, h - 1):
            self._place_wall(partition_x, y)

        # Place a door in the partition at the vertical center
        door_y = h // 2
        self._place_door(partition_x, door_y)

        # Player on the left side
        self._place_player(1, h // 2)

        # Stairs on the right side
        self._place_stairs(w - 2, h // 2)

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        """Override to handle door opening mechanics."""
        name = self.action_spec.names[action]

        # Check if player is trying to move into a door
        if name in MOVE_VECTORS:
            dx, dy = MOVE_VECTORS[name]
            nx, ny = self._player_pos[0] + dx, self._player_pos[1] + dy
            terrain = self._terrain_at(nx, ny)
            if terrain == "+":
                # Open the door: replace with floor and move player there
                self._grid[ny][nx] = "."
                self._player_pos = (nx, ny)
                self._message = "You open the door."

                # Check if this is also the goal
                if self._goal_pos and self._player_pos == self._goal_pos:
                    obs = self._render_current_observation()
                    return obs, 1.0, True, False, {"goal_reached": True}

                obs = self._render_current_observation()
                return obs, 0.0, False, False, {}

        # Default behavior for all other cases
        return super()._step(action)


class MiniHackKeyRoomS5Env(_KeyRoomBase):
    """KeyRoom with 5x5 interior."""

    _room_interior = 5

    def env_id(self) -> str:
        return "atlas_rl/minihack-keyroom-s5-v0"


class MiniHackKeyRoomS15Env(_KeyRoomBase):
    """KeyRoom with 15x15 interior."""

    _room_interior = 15

    def env_id(self) -> str:
        return "atlas_rl/minihack-keyroom-s15-v0"


class MiniHackKeyRoomDarkS5Env(_KeyRoomBase):
    """KeyRoom with 5x5 interior, dark."""

    _room_interior = 5
    _is_dark = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-keyroom-dark-s5-v0"


class MiniHackKeyRoomDarkS15Env(_KeyRoomBase):
    """KeyRoom with 15x15 interior, dark."""

    _room_interior = 15
    _is_dark = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-keyroom-dark-s15-v0"
