"""Atari Name This Game environment.

Underwater shooter. Protect your fish from enemies.

Gym ID: glyphbench/atari-namethisgame-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class NameThisGameEnv(AtariBase):
    """Name This Game: underwater shooter protecting fish.

    20x20 grid. Enemies (sharks, octopi) approach from sides.
    Player moves horizontally at surface and shoots down.
    Protect the fish at the bottom.

    Actions: NOOP, LEFT, RIGHT, FIRE
    Reward: +10 per enemy, -20 if fish eaten
    Lives: 3
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "fire downward",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _PLAYER_Y = 2
    _WATER_TOP = 4
    _FISH_Y = 17
    _MAX_BULLETS = 2

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._bullets: list[AtariEntity] = []
        self._enemies: list[AtariEntity] = []
        self._fish: list[AtariEntity] = []
        self._step_counter: int = 0
        self._spawn_cd: int = 0
        self._kills: int = 0
        self._wave_target: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-namethisgame-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._bullets = []
        self._enemies = []
        self._fish = []
        self._step_counter = 0
        self._spawn_cd = 0
        self._kills = 0
        self._wave_target = 10 + self._level * 3

        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")

        # Water surface
        for x in range(1, self._WIDTH - 1):
            self._set_cell(x, self._WATER_TOP, "~")

        # Sea floor
        for x in range(1, self._WIDTH - 1):
            self._set_cell(x, self._HEIGHT - 2, "=")

        # Fish to protect
        for i in range(3):
            fx = 5 + i * 4
            f = self._add_entity("fish", "f", fx, self._FISH_Y)
            self._fish.append(f)

        # Player at surface
        self._player_x = self._WIDTH // 2
        self._player_y = self._PLAYER_Y

        # Spawn initial enemies
        for _ in range(3):
            self._spawn_enemy()

        self._redraw()

    def _spawn_enemy(self) -> None:
        rng = self.rng
        side = int(rng.integers(0, 2))
        ex = 1 if side == 0 else self._WIDTH - 2
        ey = int(rng.integers(
            self._WATER_TOP + 1, self._HEIGHT - 3
        ))
        dx = 1 if side == 0 else -1
        etype = int(rng.integers(0, 2))
        ch = "S" if etype == 0 else "O"
        name = "shark" if etype == 0 else "octopus"
        e = self._add_entity(name, ch, ex, ey, dx=dx)
        e.data["timer"] = 0
        e.data["speed"] = max(1, 3 - self._level // 3)
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
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._player_x < self._WIDTH - 2:
            self._player_x += 1
            self._player_dir = (1, 0)

        # Fire downward
        if action_name == "FIRE" and len(self._bullets) < self._MAX_BULLETS:
            b = self._add_entity(
                "bullet", "!", self._player_x,
                self._WATER_TOP + 1, dy=1
            )
            self._bullets.append(b)

        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.y += b.dy
            if b.y >= self._HEIGHT - 2:
                b.alive = False
                continue
            for e in self._enemies:
                if e.alive and e.x == b.x and e.y == b.y:
                    e.alive = False
                    b.alive = False
                    self._on_point_scored(10)
                    reward += 10
                    self._kills += 1
                    self._message = "Enemy hit! +10"
                    break
        self._bullets = [b for b in self._bullets if b.alive]

        # Move enemies
        for e in self._enemies:
            if not e.alive:
                continue
            e.data["timer"] += 1
            spd = e.data.get("speed", 2)
            if e.data["timer"] % spd == 0:
                e.x += e.dx
                if e.x <= 0 or e.x >= self._WIDTH - 1:
                    e.dx = -e.dx
                    e.x = max(1, min(self._WIDTH - 2, e.x))

                # Drift toward fish
                if e.data["timer"] % 6 == 0:
                    if e.y < self._FISH_Y:
                        e.y += 1

        # Enemies eating fish
        for e in self._enemies:
            if not e.alive:
                continue
            for f in self._fish:
                if (
                    f.alive
                    and abs(e.x - f.x) <= 1
                    and e.y == f.y
                ):
                    f.alive = False
                    e.alive = False
                    self._score = max(0, self._score - 20)
                    reward -= 20
                    self._message = "Fish eaten! -20"
                    break

        # Check all fish dead
        alive_fish = sum(1 for f in self._fish if f.alive)
        if alive_fish == 0:
            self._on_life_lost()
            self._message = "All fish eaten! Lost a life."
            # Respawn fish
            self._fish = []
            for i in range(3):
                fx = 5 + i * 4
                f = self._add_entity("fish", "f", fx, self._FISH_Y)
                self._fish.append(f)

        # Spawn enemies
        self._spawn_cd -= 1
        alive_enemies = sum(1 for e in self._enemies if e.alive)
        if self._spawn_cd <= 0 and alive_enemies < 5:
            self._spawn_enemy()
            self._spawn_cd = max(4, 10 - self._level)

        # Wave clear
        if self._kills >= self._wave_target:
            self._on_point_scored(50)
            reward += 50
            self._message = "Wave cleared! +50"
            self._level += 1
            self._generate_level(self._level)

        self._enemies = [e for e in self._enemies if e.alive]
        self._redraw()
        info["fish_alive"] = alive_fish
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                if y == self._WATER_TOP:
                    self._set_cell(x, y, "~")
                elif y == self._HEIGHT - 2:
                    self._set_cell(x, y, "=")
                elif y < self._WATER_TOP:
                    self._set_cell(x, y, " ")
                else:
                    self._set_cell(x, y, "·")

        for f in self._fish:
            if f.alive:
                self._set_cell(f.x, f.y, "f")
        for e in self._enemies:
            if e.alive:
                self._set_cell(e.x, e.y, e.char)
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, "!")

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "wall",
            "│": "wall",
            "~": "water surface",
            "=": "sea floor",
            "·": "water",
            "S": "shark",
            "O": "octopus",
            "f": "your fish (protect)",
            "!": "your bullet",
            " ": "air",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        alive_fish = sum(1 for f in self._fish if f.alive)
        extra = (
            f"Kills: {self._kills}"
            f"/{self._wave_target}  "
            f"Fish: {alive_fish}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Move at the surface and shoot downward to destroy "
            "sharks and octopi. Protect your fish at the bottom. "
            "Clear the wave target to advance."
        )
