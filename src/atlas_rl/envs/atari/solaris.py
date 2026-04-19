"""Atari Solaris environment.

Space combat and exploration across sectors.

Gym ID: atlas_rl/atari-solaris-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec

from .base import AtariBase, AtariEntity


class SolarisEnv(AtariBase):
    """Solaris: space combat and sector exploration.

    20x20 grid. Navigate between sectors, fight enemies,
    and protect allied bases. WARP moves to adjacent sectors.

    Actions: NOOP, UP, DOWN, LEFT, RIGHT, FIRE, WARP
    Reward: +30 per enemy, +100 per sector cleared
    Lives: 3
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "UP", "DOWN", "LEFT",
            "RIGHT", "FIRE", "WARP",
        ),
        descriptions=(
            "do nothing",
            "move up",
            "move down",
            "move left",
            "move right",
            "fire forward",
            "warp to next sector",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _NUM_SECTORS = 4
    _FIRE_DX = 0
    _FIRE_DY = -1

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._enemies: list[AtariEntity] = []
        self._bullets: list[AtariEntity] = []
        self._enemy_bullets: list[AtariEntity] = []
        self._step_counter: int = 0
        self._sector: int = 0
        self._sectors_cleared: set[int] = set()
        self._fire_dx: int = 0
        self._fire_dy: int = -1
        self._bases: list[AtariEntity] = []

    def env_id(self) -> str:
        return "atlas_rl/atari-solaris-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._enemies = []
        self._bullets = []
        self._enemy_bullets = []
        self._bases = []
        self._step_counter = 0

        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "-")
            self._set_cell(x, self._HEIGHT - 1, "-")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "|")
            self._set_cell(self._WIDTH - 1, y, "|")

        rng = self.rng

        # Stars for ambiance
        for _ in range(10):
            sx = int(rng.integers(2, self._WIDTH - 2))
            sy = int(rng.integers(2, self._HEIGHT - 2))
            self._set_cell(sx, sy, ".")

        # Player
        self._player_x = self._WIDTH // 2
        self._player_y = self._HEIGHT - 4

        # Sector-specific content
        if self._sector not in self._sectors_cleared:
            num_enemies = 3 + self._level + self._sector
            for _ in range(min(num_enemies, 8)):
                ex = int(rng.integers(2, self._WIDTH - 2))
                ey = int(rng.integers(2, self._HEIGHT // 2))
                e = self._add_entity("enemy", "E", ex, ey)
                e.data["timer"] = 0
                e.data["fire_cd"] = 0
                self._enemies.append(e)

        # Allied base in some sectors
        if self._sector % 2 == 0:
            bx = int(rng.integers(5, self._WIDTH - 5))
            by = int(rng.integers(
                self._HEIGHT // 3, 2 * self._HEIGHT // 3
            ))
            base = self._add_entity("base", "B", bx, by)
            base.data["hp"] = 3
            self._bases.append(base)

        # Sector indicator
        for x in range(1, min(self._sector + 2, self._WIDTH - 1)):
            self._set_cell(x, 0, "=")

        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        # Movement
        nx, ny = self._player_x, self._player_y
        if action_name == "UP":
            ny -= 1
            self._fire_dx, self._fire_dy = 0, -1
        elif action_name == "DOWN":
            ny += 1
            self._fire_dx, self._fire_dy = 0, 1
        elif action_name == "LEFT":
            nx -= 1
            self._fire_dx, self._fire_dy = -1, 0
        elif action_name == "RIGHT":
            nx += 1
            self._fire_dx, self._fire_dy = 1, 0

        if (
            0 < nx < self._WIDTH - 1
            and 0 < ny < self._HEIGHT - 1
        ):
            self._player_x = nx
            self._player_y = ny

        # Warp to next sector
        if action_name == "WARP":
            old = self._sector
            self._sector = (
                (self._sector + 1) % self._NUM_SECTORS
            )
            self._message = (
                f"Warped: sector {old} -> {self._sector}"
            )
            self._generate_level(
                self._level * 100 + self._sector
            )
            return reward, False, info

        # Fire
        if action_name == "FIRE" and len(self._bullets) < 3:
            bx = self._player_x + self._fire_dx
            by = self._player_y + self._fire_dy
            b = self._add_entity("bullet", "*", bx, by)
            b.data["bdx"] = self._fire_dx
            b.data["bdy"] = self._fire_dy
            self._bullets.append(b)

        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.x += b.data["bdx"]
            b.y += b.data["bdy"]
            if (
                b.x <= 0
                or b.x >= self._WIDTH - 1
                or b.y <= 0
                or b.y >= self._HEIGHT - 1
            ):
                b.alive = False

        # Bullet-enemy collisions
        for b in self._bullets:
            if not b.alive:
                continue
            for e in self._enemies:
                if e.alive and e.x == b.x and e.y == b.y:
                    e.alive = False
                    b.alive = False
                    self._on_point_scored(30)
                    reward += 30
                    self._message = "Enemy destroyed! +30"
                    break

        # Enemy AI
        for e in self._enemies:
            if not e.alive:
                continue
            e.data["timer"] += 1
            if e.data["timer"] >= 3:
                e.data["timer"] = 0
                # Move toward player or base
                tx, ty = self._player_x, self._player_y
                # Sometimes target base
                if (
                    self._bases
                    and self._bases[0].alive
                    and self.rng.random() < 0.3
                ):
                    tx = self._bases[0].x
                    ty = self._bases[0].y
                dx = (
                    1 if tx > e.x
                    else (-1 if tx < e.x else 0)
                )
                dy = (
                    1 if ty > e.y
                    else (-1 if ty < e.y else 0)
                )
                if abs(dx) >= abs(dy):
                    ntx = e.x + dx
                    if 0 < ntx < self._WIDTH - 1:
                        e.x = ntx
                else:
                    nty = e.y + dy
                    if 0 < nty < self._HEIGHT - 1:
                        e.y = nty

            # Enemy fire
            e.data["fire_cd"] += 1
            if e.data["fire_cd"] >= 10:
                e.data["fire_cd"] = 0
                fdx = (
                    1
                    if self._player_x > e.x
                    else (
                        -1 if self._player_x < e.x else 0
                    )
                )
                fdy = (
                    1
                    if self._player_y > e.y
                    else (
                        -1 if self._player_y < e.y else 0
                    )
                )
                if abs(fdx) >= abs(fdy):
                    fdy = 0
                else:
                    fdx = 0
                if fdx != 0 or fdy != 0:
                    eb = self._add_entity(
                        "enemy_bullet", "o",
                        e.x + fdx, e.y + fdy,
                    )
                    eb.data["bdx"] = fdx
                    eb.data["bdy"] = fdy
                    self._enemy_bullets.append(eb)

        # Move enemy bullets
        for eb in self._enemy_bullets:
            if not eb.alive:
                continue
            eb.x += eb.data["bdx"]
            eb.y += eb.data["bdy"]
            if (
                eb.x <= 0
                or eb.x >= self._WIDTH - 1
                or eb.y <= 0
                or eb.y >= self._HEIGHT - 1
            ):
                eb.alive = False
            elif (
                eb.x == self._player_x
                and eb.y == self._player_y
            ):
                eb.alive = False
                self._on_life_lost()
                self._message = "Hit by enemy fire!"
                self._player_x = self._WIDTH // 2
                self._player_y = self._HEIGHT - 4

        # Base damage
        for base in self._bases:
            if not base.alive:
                continue
            for e in self._enemies:
                if (
                    e.alive
                    and e.x == base.x
                    and e.y == base.y
                ):
                    base.data["hp"] -= 1
                    e.alive = False
                    if base.data["hp"] <= 0:
                        base.alive = False
                        self._message = "Base destroyed!"

        # Check sector cleared
        alive_enemies = [
            e for e in self._enemies if e.alive
        ]
        if (
            len(alive_enemies) == 0
            and self._sector not in self._sectors_cleared
        ):
            self._sectors_cleared.add(self._sector)
            self._on_point_scored(100)
            reward += 100
            self._message = (
                f"Sector {self._sector} cleared! +100"
            )
            # Check all sectors cleared
            if len(self._sectors_cleared) >= self._NUM_SECTORS:
                self._level += 1
                self._sectors_cleared = set()
                self._message = "All sectors clear! Level up!"

        # Cleanup
        self._bullets = [
            b for b in self._bullets if b.alive
        ]
        self._enemy_bullets = [
            eb for eb in self._enemy_bullets if eb.alive
        ]
        self._enemies = [
            e for e in self._enemies if e.alive
        ]

        self._redraw()
        info["sector"] = self._sector
        info["sectors_cleared"] = len(self._sectors_cleared)
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")
        # Stars
        rng = self.rng
        for _ in range(5):
            sx = int(rng.integers(2, self._WIDTH - 2))
            sy = int(rng.integers(2, self._HEIGHT - 2))
            self._set_cell(sx, sy, ".")
        for e in self._enemies:
            if e.alive:
                self._set_cell(e.x, e.y, "E")
        for base in self._bases:
            if base.alive:
                self._set_cell(base.x, base.y, "B")
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, "*")
        for eb in self._enemy_bullets:
            if eb.alive:
                self._set_cell(eb.x, eb.y, "o")

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "-": "wall",
            "|": "wall",
            "=": "sector indicator",
            ".": "star",
            "E": "enemy (30pts)",
            "B": "allied base (protect!)",
            "*": "your bullet",
            "o": "enemy bullet",
            " ": "space",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Explore sectors and destroy enemies. "
            "WARP to travel between sectors. "
            "Protect allied bases (B) from enemies. "
            "Clear all sectors to advance."
        )
