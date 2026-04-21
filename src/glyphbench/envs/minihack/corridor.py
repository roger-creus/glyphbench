"""MiniHack Corridor environments.

Connected rooms in sequence with 1-cell-wide corridors between them.
Player starts in the first room and must reach the stairs in the last room.

Variants:
  - R2: 2 rooms
  - R3: 3 rooms
  - R5: 5 rooms

Each room has a 5x5 interior (7x7 including walls).

Gym IDs:
  glyphbench/minihack-corridor-r2-v0
  glyphbench/minihack-corridor-r3-v0
  glyphbench/minihack-corridor-r5-v0
"""

from __future__ import annotations

from glyphbench.envs.minihack.base import MiniHackBase


class _CorridorBase(MiniHackBase):
    """Base class for corridor environments."""

    _num_rooms: int = 2
    _room_interior: int = 5  # interior width/height of each room
    _corridor_len: int = 3  # length of corridor between rooms

    def _generate_level(self, seed: int) -> None:
        ri = self._room_interior
        cl = self._corridor_len
        nr = self._num_rooms

        # Each room is (ri + 2) wide (interior + wall on each side).
        # Between rooms there is a corridor of length cl.
        # Total width = nr * (ri + 2) + (nr - 1) * cl
        total_w = nr * (ri + 2) + (nr - 1) * cl
        total_h = ri + 2

        self._init_grid(total_w, total_h)

        # _init_grid gives us border walls and floor interior.
        # We need to fill the entire interior with walls first,
        # then carve out rooms and corridors.
        for y in range(1, total_h - 1):
            for x in range(1, total_w - 1):
                self._place_wall(x, y)

        # Carve rooms
        for i in range(nr):
            room_x_start = i * (ri + 2 + cl)
            for y in range(1, ri + 1):
                for x in range(room_x_start + 1, room_x_start + ri + 1):
                    if 1 <= x < total_w - 1:
                        self._grid[y][x] = "·"

        # Carve corridors between rooms
        corridor_y = total_h // 2
        for i in range(nr - 1):
            # End of room i (right wall interior edge)
            room_i_right = i * (ri + 2 + cl) + ri + 1
            # Start of room i+1 (left wall interior edge)
            room_next_left = (i + 1) * (ri + 2 + cl)
            # Carve from room_i_right through to room_next_left
            for x in range(room_i_right, room_next_left + 1):
                if 0 < x < total_w - 1:
                    self._grid[corridor_y][x] = "·"

        # Player in center of first room
        self._place_player(1 + ri // 2, total_h // 2)

        # Stairs in center of last room
        last_room_x = (nr - 1) * (ri + 2 + cl) + 1 + ri // 2
        self._place_stairs(min(last_room_x, total_w - 2), total_h // 2)


class MiniHackCorridorR2Env(_CorridorBase):
    """2-room corridor environment."""

    _num_rooms = 2

    def env_id(self) -> str:
        return "glyphbench/minihack-corridor-r2-v0"


class MiniHackCorridorR3Env(_CorridorBase):
    """3-room corridor environment."""

    _num_rooms = 3

    def env_id(self) -> str:
        return "glyphbench/minihack-corridor-r3-v0"


class MiniHackCorridorR5Env(_CorridorBase):
    """5-room corridor environment."""

    _num_rooms = 5

    def env_id(self) -> str:
        return "glyphbench/minihack-corridor-r5-v0"
