"""Atari Montezuma's Revenge environment.

Room-based platformer with keys, doors, enemies, and treasures.
24 rooms in a 6x4 grid layout, 30x20 per room.

Gym ID: atlas_rl/atari-montezumarevenge-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec

from .base import AtariBase


class MontezumaRevengeEnv(AtariBase):
    """Montezuma's Revenge: room-based platformer.

    The player explores a pyramid of 24 rooms (6 columns x 4 rows).
    Each room has platforms, ladders, enemies, keys, doors, and treasures.
    Collect keys to open doors, avoid enemies, grab treasures for points.

    Grid: 30 wide x 20 tall per room (with border).
    Gravity: agent falls if no platform below.
    """

    action_spec = ActionSpec(
        names=("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UP_RIGHT", "UP_LEFT"),
        descriptions=(
            "do nothing",
            "jump",
            "climb up / jump",
            "move right",
            "move left",
            "climb down / crouch",
            "jump diagonally up-right",
            "jump diagonally up-left",
        ),
    )

    _WIDTH = 30
    _HEIGHT = 20
    _ROOMS_X = 6
    _ROOMS_Y = 4
    _TOTAL_ROOMS = 24

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._lives = 5
        self._room_x: int = 0
        self._room_y: int = 0
        self._keys: set[str] = set()
        self._collected_keys: set[int] = set()
        self._collected_treasures: set[int] = set()
        self._opened_doors: set[int] = set()
        self._rooms_cleared: set[int] = set()
        self._on_ladder: bool = False
        self._jump_vy: int = 0
        self._jumping: bool = False

    def env_id(self) -> str:
        return "atlas_rl/atari-montezumarevenge-v0"

    def _room_id(self) -> int:
        return self._room_y * self._ROOMS_X + self._room_x

    def _generate_level(self, seed: int) -> None:
        self._lives = 5
        self._room_x = 0
        self._room_y = 0
        self._keys = set()
        self._collected_keys = set()
        self._collected_treasures = set()
        self._opened_doors = set()
        self._rooms_cleared = set()
        self._on_ladder = False
        self._jumping = False
        self._jump_vy = 0
        self._player_x = 2
        self._player_y = self._HEIGHT - 3
        self._build_room()

    def _build_room(self) -> None:
        """Build the current room layout."""
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        rid = self._room_id()

        # Border walls
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "#")
            self._set_cell(x, self._HEIGHT - 1, "#")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "#")
            self._set_cell(self._WIDTH - 1, y, "#")

        # Ground floor
        for x in range(1, self._WIDTH - 1):
            self._set_cell(x, self._HEIGHT - 2, "=")

        # Platforms (procedural based on room id)
        plat_seed = rid * 7 + 13
        num_platforms = 3 + (plat_seed % 3)
        for i in range(num_platforms):
            py = 4 + ((plat_seed + i * 5) % (self._HEIGHT - 8))
            px_start = 2 + ((plat_seed + i * 3) % (self._WIDTH - 10))
            px_len = 4 + ((plat_seed + i * 2) % 6)
            for x in range(px_start, min(px_start + px_len, self._WIDTH - 1)):
                self._set_cell(x, py, "=")

        # Ladders
        num_ladders = 2 + (rid % 2)
        for i in range(num_ladders):
            lx = 3 + ((plat_seed + i * 11) % (self._WIDTH - 6))
            ly_top = 3 + ((plat_seed + i * 7) % 5)
            ly_bot = ly_top + 4 + ((plat_seed + i) % 4)
            for y in range(ly_top, min(ly_bot, self._HEIGHT - 2)):
                if self._grid_at(lx, y) == " ":
                    self._set_cell(lx, y, "H")

        # Room exits (openings in walls)
        # Left exit
        if self._room_x > 0:
            for y in range(self._HEIGHT - 4, self._HEIGHT - 2):
                self._set_cell(0, y, " ")
        # Right exit
        if self._room_x < self._ROOMS_X - 1:
            for y in range(self._HEIGHT - 4, self._HEIGHT - 2):
                self._set_cell(self._WIDTH - 1, y, " ")
        # Top exit
        if self._room_y > 0:
            for x in range(self._WIDTH // 2 - 1, self._WIDTH // 2 + 2):
                self._set_cell(x, 0, " ")
        # Bottom exit
        if self._room_y < self._ROOMS_Y - 1:
            for x in range(self._WIDTH // 2 - 1, self._WIDTH // 2 + 2):
                self._set_cell(x, self._HEIGHT - 1, " ")
                self._set_cell(x, self._HEIGHT - 2, " ")

        # Enemies (skip in first room for fairness)
        if rid > 0:
            n_enemies = 1 + (rid % 3)
            for i in range(n_enemies):
                ex = 4 + ((plat_seed + i * 9) % (self._WIDTH - 8))
                ey = self._HEIGHT - 3
                edx = 1 if (i % 2 == 0) else -1
                self._add_entity("enemy", "E", ex, ey, dx=edx, dy=0)

        # Key (one per room if not already collected)
        key_color = ["red", "blue", "green", "yellow"][rid % 4]
        key_char = key_color[0].upper()
        if rid not in self._collected_keys:
            kx = 5 + ((plat_seed + 17) % (self._WIDTH - 10))
            ky = self._HEIGHT - 3
            self._add_entity("key", key_char, kx, ky)

        # Door
        if rid not in self._opened_doors:
            door_needs = ["red", "blue", "green", "yellow"][(rid + 1) % 4]
            dx_pos = self._WIDTH - 4
            dy_pos = self._HEIGHT - 4
            e = self._add_entity("door", "D", dx_pos, dy_pos)
            e.data["color"] = door_needs

        # Treasure
        if rid not in self._collected_treasures:
            tx = 10 + ((plat_seed + 23) % (self._WIDTH - 14))
            ty = self._HEIGHT - 3
            self._add_entity("treasure", "$", tx, ty)

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Movement
        dx, dy = 0, 0
        jump = False

        if action_name == "LEFT":
            dx = -1
        elif action_name == "RIGHT":
            dx = 1
        elif action_name == "UP":
            if self._on_ladder:
                dy = -1
            else:
                jump = True
        elif action_name == "DOWN":
            dy = 1  # climb down or crouch
        elif action_name == "FIRE":
            jump = True
        elif action_name == "UP_RIGHT":
            dx = 1
            jump = True
        elif action_name == "UP_LEFT":
            dx = -1
            jump = True

        # Handle jumping
        if jump and not self._jumping and not self._on_ladder:
            self._jumping = True
            self._jump_vy = -2

        if self._jumping:
            self._jump_vy += 1  # gravity
            dy = self._jump_vy
            if self._jump_vy >= 0:
                # Check landing
                land_y = self._player_y + dy
                if land_y >= self._HEIGHT - 3 or self._is_platform(self._player_x, land_y + 1):
                    dy = 0
                    # Find exact landing
                    for check_y in range(self._player_y, min(land_y + 1, self._HEIGHT - 2)):
                        if self._is_platform(self._player_x, check_y + 1):
                            dy = check_y - self._player_y
                            break
                    self._jumping = False
                    self._jump_vy = 0

        # Check ladder
        new_x = self._player_x + dx
        new_y = self._player_y + dy
        self._on_ladder = self._grid_at(self._player_x, self._player_y) == "H"

        # Bounds and collision
        if 0 <= new_x < self._WIDTH and 0 <= new_y < self._HEIGHT:
            cell = self._grid_at(new_x, new_y)
            if cell != "#":
                self._player_x = new_x
                self._player_y = new_y

        # Gravity (if not jumping, not on ladder, not on platform)
        if not self._jumping and not self._on_ladder:
            below = self._player_y + 1
            if (
                below < self._HEIGHT
                and not self._is_platform(self._player_x, below)
                and self._grid_at(self._player_x, below) != "#"
            ):
                self._player_y = below

        # Update on_ladder status
        self._on_ladder = self._grid_at(self._player_x, self._player_y) == "H"

        # Room transitions
        if self._player_x <= 0 and self._room_x > 0:
            self._room_x -= 1
            self._player_x = self._WIDTH - 2
            self._build_room()
        elif self._player_x >= self._WIDTH - 1 and self._room_x < self._ROOMS_X - 1:
            self._room_x += 1
            self._player_x = 1
            self._build_room()
        elif self._player_y <= 0 and self._room_y > 0:
            self._room_y -= 1
            self._player_y = self._HEIGHT - 3
            self._build_room()
        elif self._player_y >= self._HEIGHT - 1 and self._room_y < self._ROOMS_Y - 1:
            self._room_y += 1
            self._player_y = 2
            self._build_room()

        # Entity interactions
        rid = self._room_id()
        for e in self._entities:
            if not e.alive:
                continue
            if e.x == self._player_x and e.y == self._player_y:
                if e.etype == "enemy":
                    self._on_life_lost()
                    self._player_x = 2
                    self._player_y = self._HEIGHT - 3
                    self._message = "Hit by enemy!"
                    break
                elif e.etype == "key":
                    color = e.char
                    self._keys.add(color)
                    e.alive = False
                    self._collected_keys.add(rid)
                    self._message = f"Got key {color}!"
                elif e.etype == "treasure":
                    e.alive = False
                    self._collected_treasures.add(rid)
                    self._on_point_scored(100)
                    reward += 100
                    self._message = "Treasure! +100"
                elif e.etype == "door":
                    needed = e.data.get("color", "red")
                    needed_char = needed[0].upper()
                    if needed_char in self._keys:
                        e.alive = False
                        self._opened_doors.add(rid)
                        self._keys.discard(needed_char)
                        self._message = "Door opened!"
                    else:
                        # Block passage
                        self._player_x -= 1
                        self._message = f"Need {needed} key!"

        # Enemy bounce logic
        for e in self._entities:
            if e.etype == "enemy" and e.alive:
                next_x = e.x + e.dx
                if next_x <= 1 or next_x >= self._WIDTH - 2:
                    e.dx = -e.dx

        # Check room cleared
        treasures_left = any(e.alive and e.etype == "treasure" for e in self._entities)
        enemies_left = any(e.alive and e.etype == "enemy" for e in self._entities)
        if not treasures_left and not enemies_left and rid not in self._rooms_cleared:
            self._rooms_cleared.add(rid)
            self._on_point_scored(300)
            reward += 300
            self._message = "Room cleared! +300"

        info["room"] = (self._room_x, self._room_y)
        info["keys"] = list(self._keys)
        info["rooms_cleared"] = len(self._rooms_cleared)

        return reward, self._game_over, info

    def _is_platform(self, x: int, y: int) -> bool:
        ch = self._grid_at(x, y)
        return ch in ("=", "#")

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "#": "wall",
            "=": "platform",
            "H": "ladder",
            " ": "empty",
            "D": "door",
            "$": "treasure",
            "E": "enemy",
            "R": "red key",
            "B": "blue key",
            "G": "green key",
            "Y": "yellow key",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Explore the pyramid rooms. Collect keys to open doors. "
            "Grab treasures for points. Avoid enemies. "
            "Clear rooms by collecting all treasures and defeating enemies."
        )
