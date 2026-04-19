"""Atari Gravitar environment.

Space shooter with gravity and fuel management.

Gym ID: atlas_rl/atari-gravitar-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec

from .base import AtariBase, AtariEntity


class GravitarEnv(AtariBase):
    """Gravitar: space shooter with gravity.

    20x20 grid. Navigate planets with gravity, shoot enemies,
    collect fuel. Multiple planets accessible via a sector map.

    Actions: NOOP, LEFT, RIGHT, THRUST, FIRE
    Reward: +30 per enemy, +10 per fuel pickup
    Lives: 3
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "THRUST", "FIRE"),
        descriptions=(
            "do nothing",
            "rotate left",
            "rotate right",
            "thrust against gravity",
            "fire in facing direction",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _MAX_FUEL = 120
    _DIRS = (
        (0, -1), (1, -1), (1, 0), (1, 1),
        (0, 1), (-1, 1), (-1, 0), (-1, -1),
    )

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._facing: int = 0
        self._fuel: int = self._MAX_FUEL
        self._enemies: list[AtariEntity] = []
        self._bullets: list[AtariEntity] = []
        self._enemy_bullets: list[AtariEntity] = []
        self._fuel_pickups: list[AtariEntity] = []
        self._terrain: set[tuple[int, int]] = set()
        self._step_counter: int = 0
        self._gravity_timer: int = 0
        self._planet: int = 0

    def env_id(self) -> str:
        return "atlas_rl/atari-gravitar-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._enemies = []
        self._bullets = []
        self._enemy_bullets = []
        self._fuel_pickups = []
        self._terrain = set()
        self._step_counter = 0
        self._gravity_timer = 0
        self._facing = 0
        self._fuel = self._MAX_FUEL

        rng = self.rng

        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "-")
            self._set_cell(x, self._HEIGHT - 1, "-")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "|")
            self._set_cell(self._WIDTH - 1, y, "|")

        # Terrain (planet surface at bottom area)
        base_y = self._HEIGHT - 4
        for x in range(1, self._WIDTH - 1):
            variation = int(rng.integers(-2, 3))
            ty = base_y + variation
            ty = max(self._HEIGHT // 2, min(ty, self._HEIGHT - 2))
            for y in range(ty, self._HEIGHT - 1):
                self._terrain.add((x, y))
                self._set_cell(x, y, "=")

        # Enemies on terrain surface
        num_enemies = 3 + self._level
        for _ in range(min(num_enemies, 8)):
            for _ in range(20):
                ex = int(rng.integers(2, self._WIDTH - 2))
                # Place on terrain surface
                ey = base_y - 1
                for scan_y in range(
                    self._HEIGHT // 2, self._HEIGHT - 1
                ):
                    if (ex, scan_y) in self._terrain:
                        ey = scan_y - 1
                        break
                if (
                    ey > 0
                    and (ex, ey) not in self._terrain
                ):
                    e = self._add_entity(
                        "enemy", "E", ex, ey
                    )
                    e.data["fire_cd"] = 0
                    self._enemies.append(e)
                    break

        # Fuel pickups
        for _ in range(3):
            for _ in range(20):
                fx = int(rng.integers(2, self._WIDTH - 2))
                fy = int(
                    rng.integers(3, self._HEIGHT // 2)
                )
                if (fx, fy) not in self._terrain:
                    fp = self._add_entity(
                        "fuel_pickup", "F", fx, fy
                    )
                    self._fuel_pickups.append(fp)
                    break

        # Player at top
        self._player_x = self._WIDTH // 2
        self._player_y = 3

        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        # Rotate
        if action_name == "LEFT":
            self._facing = (self._facing - 1) % 8
        elif action_name == "RIGHT":
            self._facing = (self._facing + 1) % 8

        # Thrust (costs fuel)
        if action_name == "THRUST" and self._fuel > 0:
            self._fuel -= 2
            dx, dy = self._DIRS[self._facing]
            nx = self._player_x + dx
            ny = self._player_y + dy
            if (
                0 < nx < self._WIDTH - 1
                and 0 < ny < self._HEIGHT - 1
                and (nx, ny) not in self._terrain
            ):
                self._player_x = nx
                self._player_y = ny

        # Gravity pulls down every 3 steps
        self._gravity_timer += 1
        if self._gravity_timer >= 3:
            self._gravity_timer = 0
            gy = self._player_y + 1
            if (
                0 < gy < self._HEIGHT - 1
                and (self._player_x, gy) not in self._terrain
            ):
                self._player_y = gy
            elif (
                self._player_x, gy
            ) in self._terrain:
                # Crash into terrain
                self._on_life_lost()
                self._message = "Crashed into surface!"
                self._player_x = self._WIDTH // 2
                self._player_y = 3
                self._fuel = self._MAX_FUEL

        # Fire
        if action_name == "FIRE" and len(self._bullets) < 3:
            dx, dy = self._DIRS[self._facing]
            bx = self._player_x + dx
            by = self._player_y + dy
            b = self._add_entity("bullet", "*", bx, by)
            b.data["bdx"] = dx
            b.data["bdy"] = dy
            b.data["life"] = 10
            self._bullets.append(b)

        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.data["life"] -= 1
            if b.data["life"] <= 0:
                b.alive = False
                continue
            b.x += b.data["bdx"]
            b.y += b.data["bdy"]
            if (
                b.x <= 0
                or b.x >= self._WIDTH - 1
                or b.y <= 0
                or b.y >= self._HEIGHT - 1
                or (b.x, b.y) in self._terrain
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
                    self._message = "Enemy down! +30"
                    break

        # Fuel pickup collection
        for fp in self._fuel_pickups:
            if (
                fp.alive
                and fp.x == self._player_x
                and fp.y == self._player_y
            ):
                fp.alive = False
                self._fuel = min(
                    self._MAX_FUEL, self._fuel + 40
                )
                self._on_point_scored(10)
                reward += 10
                self._message = "Fuel +40! +10pts"

        # Enemy fire
        for e in self._enemies:
            if not e.alive:
                continue
            e.data["fire_cd"] += 1
            if e.data["fire_cd"] >= 12:
                e.data["fire_cd"] = 0
                fdx = (
                    1
                    if self._player_x > e.x
                    else (-1 if self._player_x < e.x else 0)
                )
                fdy = (
                    1
                    if self._player_y > e.y
                    else (-1 if self._player_y < e.y else 0)
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
                    eb.data["life"] = 8
                    self._enemy_bullets.append(eb)

        # Move enemy bullets
        for eb in self._enemy_bullets:
            if not eb.alive:
                continue
            eb.data["life"] -= 1
            if eb.data["life"] <= 0:
                eb.alive = False
                continue
            eb.x += eb.data["bdx"]
            eb.y += eb.data["bdy"]
            if (
                eb.x <= 0
                or eb.x >= self._WIDTH - 1
                or eb.y <= 0
                or eb.y >= self._HEIGHT - 1
                or (eb.x, eb.y) in self._terrain
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
                self._player_y = 3

        # Out of fuel
        if self._fuel <= 0:
            self._on_life_lost()
            self._message = "Out of fuel!"
            self._fuel = self._MAX_FUEL
            self._player_x = self._WIDTH // 2
            self._player_y = 3

        # Level clear
        alive_enemies = [
            e for e in self._enemies if e.alive
        ]
        if len(alive_enemies) == 0:
            self._level += 1
            self._planet += 1
            self._message = (
                f"Planet {self._planet} cleared!"
            )
            self._generate_level(
                self._level + self._planet
            )

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
        self._fuel_pickups = [
            f for f in self._fuel_pickups if f.alive
        ]

        self._redraw()
        info["fuel"] = self._fuel
        info["planet"] = self._planet
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                if (x, y) in self._terrain:
                    self._set_cell(x, y, "=")
                else:
                    self._set_cell(x, y, " ")
        for e in self._enemies:
            if e.alive:
                self._set_cell(e.x, e.y, e.char)
        for fp in self._fuel_pickups:
            if fp.alive:
                self._set_cell(fp.x, fp.y, "F")
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
            "=": "terrain",
            "E": "enemy (30pts)",
            "F": "fuel pickup (+40 fuel)",
            "*": "your bullet",
            "o": "enemy bullet",
            " ": "space",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Navigate with gravity pulling you down. "
            "THRUST to fly, LEFT/RIGHT to rotate, FIRE to shoot. "
            "Destroy enemies, collect fuel (F). "
            "Crashing into terrain or running out of fuel costs a life."
        )
