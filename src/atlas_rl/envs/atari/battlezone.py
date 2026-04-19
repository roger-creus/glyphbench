"""Atari BattleZone environment.

Top-down tank combat. Destroy enemy tanks on a battlefield.

Gym ID: atlas_rl/atari-battlezone-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec

from .base import AtariBase, AtariEntity


class BattleZoneEnv(AtariBase):
    """BattleZone: top-down tank combat.

    20x20 grid. Destroy enemy tanks that spawn and pursue you.
    Obstacles provide cover. Enemies fire back.

    Actions: NOOP, LEFT, RIGHT, UP, DOWN, FIRE
    Reward: +100 per enemy tank destroyed
    Lives: 3
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "move up",
            "move down",
            "fire in last moved direction",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _MAX_ENEMIES = 3
    _SPAWN_INTERVAL = 20

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._tanks: list[AtariEntity] = []
        self._bullets: list[AtariEntity] = []
        self._enemy_bullets: list[AtariEntity] = []
        self._step_counter: int = 0
        self._fire_dx: int = 0
        self._fire_dy: int = -1
        self._obstacles: set[tuple[int, int]] = set()

    def env_id(self) -> str:
        return "atlas_rl/atari-battlezone-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._tanks = []
        self._bullets = []
        self._enemy_bullets = []
        self._step_counter = 0
        self._obstacles = set()

        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "-")
            self._set_cell(x, self._HEIGHT - 1, "-")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "|")
            self._set_cell(self._WIDTH - 1, y, "|")

        # Place obstacles
        rng = self.rng
        num_obs = 6 + self._level
        for _ in range(min(num_obs, 15)):
            ox = int(rng.integers(3, self._WIDTH - 3))
            oy = int(rng.integers(3, self._HEIGHT - 3))
            self._obstacles.add((ox, oy))
            self._set_cell(ox, oy, "#")

        # Player at center bottom
        self._player_x = self._WIDTH // 2
        self._player_y = self._HEIGHT - 3

        # Spawn initial enemies
        for _ in range(min(2, self._MAX_ENEMIES)):
            self._spawn_enemy()

        self._redraw()

    def _spawn_enemy(self) -> None:
        rng = self.rng
        for _ in range(20):
            ex = int(rng.integers(2, self._WIDTH - 2))
            ey = int(rng.integers(2, self._HEIGHT // 2))
            if (ex, ey) not in self._obstacles:
                dist = abs(ex - self._player_x) + abs(
                    ey - self._player_y
                )
                if dist > 5:
                    t = self._add_entity("tank", "T", ex, ey)
                    t.data["timer"] = 0
                    t.data["fire_timer"] = 0
                    self._tanks.append(t)
                    return

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        # Move player
        nx, ny = self._player_x, self._player_y
        if action_name == "LEFT":
            nx -= 1
            self._fire_dx, self._fire_dy = -1, 0
        elif action_name == "RIGHT":
            nx += 1
            self._fire_dx, self._fire_dy = 1, 0
        elif action_name == "UP":
            ny -= 1
            self._fire_dx, self._fire_dy = 0, -1
        elif action_name == "DOWN":
            ny += 1
            self._fire_dx, self._fire_dy = 0, 1

        if (
            0 < nx < self._WIDTH - 1
            and 0 < ny < self._HEIGHT - 1
            and (nx, ny) not in self._obstacles
        ):
            self._player_x = nx
            self._player_y = ny

        # Fire
        if action_name == "FIRE" and len(self._bullets) < 3:
            bx = self._player_x + self._fire_dx
            by = self._player_y + self._fire_dy
            b = self._add_entity("bullet", "*", bx, by)
            b.data["bdx"] = self._fire_dx
            b.data["bdy"] = self._fire_dy
            self._bullets.append(b)

        # Move player bullets
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
                or (b.x, b.y) in self._obstacles
            ):
                b.alive = False

        # Bullet-tank collision
        for b in self._bullets:
            if not b.alive:
                continue
            for t in self._tanks:
                if t.alive and t.x == b.x and t.y == b.y:
                    t.alive = False
                    b.alive = False
                    self._on_point_scored(100)
                    reward += 100
                    self._message = "Tank destroyed! +100"
                    break

        # Move enemy tanks toward player
        for t in self._tanks:
            if not t.alive:
                continue
            t.data["timer"] += 1
            if t.data["timer"] >= 3:
                t.data["timer"] = 0
                dx = (
                    1
                    if self._player_x > t.x
                    else (-1 if self._player_x < t.x else 0)
                )
                dy = (
                    1
                    if self._player_y > t.y
                    else (-1 if self._player_y < t.y else 0)
                )
                # Prefer axis with larger distance
                if abs(self._player_x - t.x) >= abs(
                    self._player_y - t.y
                ):
                    ntx, nty = t.x + dx, t.y
                else:
                    ntx, nty = t.x, t.y + dy
                if (
                    0 < ntx < self._WIDTH - 1
                    and 0 < nty < self._HEIGHT - 1
                    and (ntx, nty) not in self._obstacles
                ):
                    t.x = ntx
                    t.y = nty

            # Enemy fire
            t.data["fire_timer"] += 1
            if t.data["fire_timer"] >= 8:
                t.data["fire_timer"] = 0
                fdx = (
                    1
                    if self._player_x > t.x
                    else (-1 if self._player_x < t.x else 0)
                )
                fdy = (
                    1
                    if self._player_y > t.y
                    else (-1 if self._player_y < t.y else 0)
                )
                if fdx != 0 or fdy != 0:
                    # Pick dominant axis
                    if abs(fdx) >= abs(fdy):
                        fdy = 0
                    else:
                        fdx = 0
                    eb = self._add_entity(
                        "enemy_bullet", "o", t.x + fdx, t.y + fdy
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
                or (eb.x, eb.y) in self._obstacles
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
                self._player_y = self._HEIGHT - 3

        # Player-tank collision
        for t in self._tanks:
            if (
                t.alive
                and t.x == self._player_x
                and t.y == self._player_y
            ):
                t.alive = False
                self._on_life_lost()
                self._message = "Rammed by tank!"
                self._player_x = self._WIDTH // 2
                self._player_y = self._HEIGHT - 3

        # Spawn new enemies
        alive_tanks = [t for t in self._tanks if t.alive]
        if (
            len(alive_tanks) < self._MAX_ENEMIES
            and self._step_counter % self._SPAWN_INTERVAL == 0
        ):
            self._spawn_enemy()

        # Cleanup
        self._bullets = [b for b in self._bullets if b.alive]
        self._enemy_bullets = [
            eb for eb in self._enemy_bullets if eb.alive
        ]
        self._tanks = [t for t in self._tanks if t.alive]

        self._redraw()
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                if (x, y) in self._obstacles:
                    self._set_cell(x, y, "#")
                else:
                    self._set_cell(x, y, " ")
        for t in self._tanks:
            if t.alive:
                self._set_cell(t.x, t.y, "T")
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
            "#": "obstacle",
            "T": "enemy tank (100pts)",
            "*": "your bullet",
            "o": "enemy bullet",
            " ": "ground",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Destroy enemy tanks on the battlefield. "
            "Move with UP/DOWN/LEFT/RIGHT, FIRE to shoot. "
            "Use obstacles (#) for cover. "
            "Enemy tanks pursue you and fire back."
        )
