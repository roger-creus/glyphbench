"""Atari Star Gunner environment.

Side-scrolling space shooter with waves of enemies.

Gym ID: atlas_rl/atari-stargunner-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec

from .base import AtariBase, AtariEntity


class StarGunnerEnv(AtariBase):
    """Star Gunner: side-scrolling space shooter.

    30x16 grid. Enemies scroll in from the right.
    Shoot them before they pass or hit you.

    Actions: NOOP, LEFT, RIGHT, UP, DOWN, FIRE
    Reward: +10 per enemy, +5 bonus item
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
            "fire a bullet right",
        ),
    )

    _WIDTH = 30
    _HEIGHT = 16
    _MAX_BULLETS = 4

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._bullets: list[AtariEntity] = []
        self._enemies: list[AtariEntity] = []
        self._enemy_bullets: list[AtariEntity] = []
        self._step_counter: int = 0
        self._spawn_cd: int = 0
        self._enemies_killed: int = 0
        self._wave_target: int = 0

    def env_id(self) -> str:
        return "atlas_rl/atari-stargunner-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._bullets = []
        self._enemies = []
        self._enemy_bullets = []
        self._step_counter = 0
        self._spawn_cd = 0
        self._enemies_killed = 0
        self._wave_target = 8 + self._level * 2

        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "-")
            self._set_cell(x, self._HEIGHT - 1, "-")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "|")
            self._set_cell(self._WIDTH - 1, y, "|")

        # Stars in background
        rng = self.rng
        for _ in range(10):
            sx = int(rng.integers(1, self._WIDTH - 1))
            sy = int(rng.integers(1, self._HEIGHT - 1))
            self._set_cell(sx, sy, "*")

        self._player_x = 3
        self._player_y = self._HEIGHT // 2
        self._redraw()

    def _spawn_enemy(self) -> None:
        rng = self.rng
        ey = int(rng.integers(2, self._HEIGHT - 2))
        etype = int(rng.integers(0, 3))
        chars = ("V", "W", "X")
        speeds = (-1, -1, -2)
        e = self._add_entity(
            "enemy", chars[etype], self._WIDTH - 2, ey,
            dx=speeds[etype]
        )
        e.data["timer"] = 0
        e.data["fire_cd"] = int(rng.integers(8, 20))
        e.data["pts"] = (etype + 1) * 10
        self._enemies.append(e)

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        # Move player
        if action_name == "LEFT" and self._player_x > 1:
            self._player_x -= 1
        elif action_name == "RIGHT" and self._player_x < self._WIDTH // 2:
            self._player_x += 1
        elif action_name == "UP" and self._player_y > 1:
            self._player_y -= 1
        elif action_name == "DOWN" and self._player_y < self._HEIGHT - 2:
            self._player_y += 1

        # Fire
        if action_name == "FIRE" and len(self._bullets) < self._MAX_BULLETS:
            b = self._add_entity(
                "bullet", ">", self._player_x + 1,
                self._player_y, dx=2
            )
            self._bullets.append(b)

        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.x += b.dx
            if b.x >= self._WIDTH - 1:
                b.alive = False
                continue
            for e in self._enemies:
                if e.alive and e.x == b.x and e.y == b.y:
                    e.alive = False
                    b.alive = False
                    pts = e.data.get("pts", 10)
                    self._on_point_scored(pts)
                    reward += pts
                    self._enemies_killed += 1
                    self._message = f"Enemy destroyed! +{pts}"
                    break
        self._bullets = [b for b in self._bullets if b.alive]

        # Move enemies
        for e in self._enemies:
            if not e.alive:
                continue
            e.data["timer"] += 1
            if e.data["timer"] % 2 == 0:
                e.x += e.dx
                if e.x <= 0:
                    e.alive = False
                    continue

            # Enemy fire
            if e.data["timer"] % e.data["fire_cd"] == 0:
                if len(self._enemy_bullets) < 3:
                    eb = self._add_entity(
                        "ebullet", "<", e.x - 1, e.y, dx=-1
                    )
                    self._enemy_bullets.append(eb)

        # Move enemy bullets
        for eb in self._enemy_bullets:
            if not eb.alive:
                continue
            eb.x += eb.dx
            if eb.x <= 0:
                eb.alive = False
                continue
            if eb.x == self._player_x and eb.y == self._player_y:
                eb.alive = False
                self._on_life_lost()
                self._message = "Hit! Lost a life."
                self._player_x = 3
                self._player_y = self._HEIGHT // 2
        self._enemy_bullets = [
            eb for eb in self._enemy_bullets if eb.alive
        ]

        # Player collision with enemy
        for e in self._enemies:
            if (
                e.alive
                and e.x == self._player_x
                and e.y == self._player_y
            ):
                e.alive = False
                self._on_life_lost()
                self._message = "Collision!"
                self._player_x = 3
                self._player_y = self._HEIGHT // 2
                break

        # Spawn enemies
        self._spawn_cd -= 1
        alive_enemies = [e for e in self._enemies if e.alive]
        total_spawned = self._enemies_killed + len(alive_enemies)
        if (
            self._spawn_cd <= 0
            and total_spawned < self._wave_target
            and len(alive_enemies) < 6
        ):
            self._spawn_enemy()
            self._spawn_cd = max(5, 12 - self._level)

        # Level clear
        if self._enemies_killed >= self._wave_target:
            self._on_point_scored(50)
            reward += 50
            self._message = "Wave cleared! +50"
            self._level += 1
            self._generate_level(self._level)

        self._enemies = [e for e in self._enemies if e.alive]
        self._redraw()
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")
        # Redraw stars
        rng_star = self.rng
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                if (x * 7 + y * 13) % 37 == 0:
                    self._set_cell(x, y, "*")

        for e in self._enemies:
            if e.alive:
                self._set_cell(e.x, e.y, e.char)
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, ">")
        for eb in self._enemy_bullets:
            if eb.alive:
                self._set_cell(eb.x, eb.y, "<")

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "-": "wall",
            "|": "wall",
            "*": "star",
            "V": "enemy fighter (10pts)",
            "W": "enemy bomber (20pts)",
            "X": "enemy ace (30pts)",
            ">": "your bullet",
            "<": "enemy bullet",
            " ": "space",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Shoot enemies scrolling in from the right. "
            "Dodge enemy fire and collisions. "
            "Destroy all wave targets to advance."
        )
