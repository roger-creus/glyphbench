"""Atari Kangaroo environment.

Vertical platformer. Kangaroo climbs platforms, punches monkeys,
and rescues baby at the top.

Gym ID: atlas_rl/atari-kangaroo-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec

from .base import AtariBase


class KangarooEnv(AtariBase):
    """Kangaroo: vertical platformer.

    The player (kangaroo) must climb upward through platforms
    to reach the baby kangaroo at the top. Punch monkeys, collect
    fruit, avoid falling coconuts.

    Grid: 20 wide x 20 tall.
    Gravity: agent falls if no platform below.
    """

    action_spec = ActionSpec(
        names=("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN"),
        descriptions=(
            "do nothing",
            "punch",
            "jump up",
            "move right",
            "move left",
            "duck / climb down",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._lives = 3
        self._jumping: bool = False
        self._jump_vy: int = 0
        self._fruits_collected: int = 0

    def env_id(self) -> str:
        return "atlas_rl/atari-kangaroo-v0"

    def _generate_level(self, seed: int) -> None:
        self._jumping = False
        self._jump_vy = 0
        self._fruits_collected = 0
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []

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

        # Platforms at various heights (4 levels)
        platform_configs = [
            (self._HEIGHT - 6, 2, 9),    # level 1 left
            (self._HEIGHT - 6, 11, 18),   # level 1 right
            (self._HEIGHT - 10, 4, 16),   # level 2
            (self._HEIGHT - 14, 2, 10),   # level 3 left
            (self._HEIGHT - 14, 12, 18),  # level 3 right
            (3, 5, 15),                   # top level (baby)
        ]
        for py, x_start, x_end in platform_configs:
            if py > 0:
                for x in range(x_start, min(x_end, self._WIDTH - 1)):
                    self._set_cell(x, py, "=")

        # Ladders connecting platforms
        ladder_positions = [
            (10, self._HEIGHT - 6, self._HEIGHT - 2),
            (6, self._HEIGHT - 10, self._HEIGHT - 6),
            (14, self._HEIGHT - 10, self._HEIGHT - 6),
            (8, self._HEIGHT - 14, self._HEIGHT - 10),
            (10, 3, self._HEIGHT - 14),
        ]
        for lx, ly_top, ly_bot in ladder_positions:
            for y in range(ly_top, ly_bot):
                if (
                    0 < lx < self._WIDTH - 1
                    and 0 < y < self._HEIGHT - 1
                    and self._grid_at(lx, y) == " "
                ):
                    self._set_cell(lx, y, "H")

        # Baby kangaroo at top
        self._add_entity("baby", "B", self._WIDTH // 2, 2)

        # Fruits on platforms
        fruit_positions = [
            (5, self._HEIGHT - 3),
            (15, self._HEIGHT - 7),
            (8, self._HEIGHT - 11),
            (14, self._HEIGHT - 15),
        ]
        for fx, fy in fruit_positions:
            if 0 < fx < self._WIDTH - 1 and 0 < fy < self._HEIGHT - 1:
                self._add_entity("fruit", "F", fx, fy)

        # Monkeys on platforms
        monkey_positions = [
            (4, self._HEIGHT - 3, 1),
            (12, self._HEIGHT - 7, -1),
            (6, self._HEIGHT - 11, 1),
        ]
        for mx, my, mdx in monkey_positions:
            self._add_entity("enemy", "M", mx, my, dx=mdx, dy=0)

        # Player starts at bottom
        self._player_x = 2
        self._player_y = self._HEIGHT - 3

    def _is_platform(self, x: int, y: int) -> bool:
        ch = self._grid_at(x, y)
        return ch in ("=", "#")

    def _on_ladder(self) -> bool:
        return self._grid_at(self._player_x, self._player_y) == "H"

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        dx, dy = 0, 0
        jump = False
        punch = False

        if action_name == "LEFT":
            dx = -1
        elif action_name == "RIGHT":
            dx = 1
        elif action_name == "UP":
            if self._on_ladder():
                dy = -1
            else:
                jump = True
        elif action_name == "DOWN":
            if self._on_ladder():
                dy = 1
        elif action_name == "FIRE":
            punch = True

        # Jumping
        if jump and not self._jumping and not self._on_ladder():
            self._jumping = True
            self._jump_vy = -2

        if self._jumping:
            self._jump_vy += 1
            dy = self._jump_vy
            if self._jump_vy >= 0:
                land_y = self._player_y + dy
                for check_y in range(self._player_y + 1, min(land_y + 1, self._HEIGHT - 1)):
                    if self._is_platform(self._player_x, check_y):
                        dy = check_y - 1 - self._player_y
                        self._jumping = False
                        self._jump_vy = 0
                        break
                else:
                    if land_y >= self._HEIGHT - 2:
                        dy = self._HEIGHT - 3 - self._player_y
                        self._jumping = False
                        self._jump_vy = 0

        # Move
        new_x = self._player_x + dx
        new_y = self._player_y + dy

        if 0 < new_x < self._WIDTH - 1:
            if not self._is_platform(new_x, self._player_y):
                self._player_x = new_x
            elif self._is_platform(new_x, self._player_y) and self._on_ladder():
                pass  # blocked by platform while on ladder horizontally
            else:
                pass  # blocked
        if 0 < new_y < self._HEIGHT - 1:
            cell = self._grid_at(self._player_x, new_y)
            if cell != "#":
                self._player_y = new_y

        # Gravity
        if not self._jumping and not self._on_ladder():
            below = self._player_y + 1
            if below < self._HEIGHT and not self._is_platform(self._player_x, below):
                self._player_y = below

        # Punch: destroy nearby enemies
        if punch:
            for e in self._entities:
                if (
                    e.etype == "enemy"
                    and e.alive
                    and abs(e.x - self._player_x) <= 1
                    and e.y == self._player_y
                ):
                    e.alive = False
                    self._on_point_scored(50)
                    reward += 50
                    self._message = "Punched monkey! +50"

        # Enemy bounce
        for e in self._entities:
            if e.etype == "enemy" and e.alive:
                nx = e.x + e.dx
                if nx <= 1 or nx >= self._WIDTH - 2 or self._is_platform(nx, e.y):
                    e.dx = -e.dx

        # Entity collisions
        for e in self._entities:
            if not e.alive:
                continue
            if e.x == self._player_x and e.y == self._player_y:
                if e.etype == "enemy":
                    self._on_life_lost()
                    self._player_x = 2
                    self._player_y = self._HEIGHT - 3
                    self._jumping = False
                    self._jump_vy = 0
                    self._message = "Hit by monkey!"
                    break
                elif e.etype == "fruit":
                    e.alive = False
                    self._fruits_collected += 1
                    self._on_point_scored(100)
                    reward += 100
                    self._message = "Fruit! +100"
                elif e.etype == "baby":
                    e.alive = False
                    self._on_point_scored(200)
                    reward += 200
                    self._message = "Baby rescued! +200"
                    # Next level
                    self._level += 1
                    self._generate_level(self._level)

        info["fruits"] = self._fruits_collected

        return reward, self._game_over, info

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "#": "wall",
            "=": "platform",
            "H": "ladder",
            " ": "empty",
            "F": "fruit",
            "M": "monkey",
            "B": "baby kangaroo",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Climb the platforms to rescue the baby kangaroo at the top. "
            "Collect fruit for bonus points. Punch monkeys with FIRE."
        )
