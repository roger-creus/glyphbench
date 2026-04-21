"""Atari Seaquest environment.

Submarine shooter with oxygen management.

Gym ID: glyphbench/atari-seaquest-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec

from glyphbench.core.ascii_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class SeaquestEnv(AtariBase):
    """Seaquest: submarine combat with oxygen management.

    20x20 grid. Shoot enemies, rescue divers, surface for
    oxygen. Running out of oxygen kills you.

    Actions: NOOP, UP, DOWN, LEFT, RIGHT, FIRE, SURFACE
    Reward: +10 per enemy, +50 per diver rescued
    Lives: 3
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "UP", "DOWN", "LEFT", "RIGHT",
            "FIRE", "SURFACE",
        ),
        descriptions=(
            "do nothing",
            "move up",
            "move down",
            "move left",
            "move right",
            "fire torpedo in facing direction",
            "surface to refill oxygen (must be at top)",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _SURFACE_Y = 2
    _SEABED_Y = 18

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._enemies: list[AtariEntity] = []
        self._divers: list[AtariEntity] = []
        self._torpedoes: list[AtariEntity] = []
        self._enemy_torpedoes: list[AtariEntity] = []
        self._step_counter: int = 0
        self._oxygen: int = 100
        self._facing: int = 1
        self._rescued: int = 0
        self._carried: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-seaquest-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._enemies = []
        self._divers = []
        self._torpedoes = []
        self._enemy_torpedoes = []
        self._step_counter = 0
        self._oxygen = 100
        self._carried = 0

        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, 1, "~")
            self._set_cell(x, self._HEIGHT - 1, "=")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")

        rng = self.rng
        # Spawn enemies (fish/subs)
        n_enemies = min(3 + self._level, 8)
        for i in range(n_enemies):
            ey = int(
                rng.integers(self._SURFACE_Y + 1, self._SEABED_Y)
            )
            if rng.random() < 0.5:
                ex = 1
                edx = 1
                ch = "→"
            else:
                ex = self._WIDTH - 2
                edx = -1
                ch = "←"
            e = self._add_entity("enemy", ch, ex, ey, dx=edx)
            e.data["shoots"] = i % 3 == 0
            self._enemies.append(e)

        # Spawn divers
        n_divers = min(2 + self._level, 5)
        for _ in range(n_divers):
            dy = int(
                rng.integers(
                    self._SURFACE_Y + 3, self._SEABED_Y,
                )
            )
            dx_pos = int(rng.integers(2, self._WIDTH - 2))
            d = self._add_entity("diver", "D", dx_pos, dy)
            d.data["dir"] = int(rng.choice([-1, 1]))
            self._divers.append(d)

        self._player_x = self._WIDTH // 2
        self._player_y = self._SURFACE_Y + 3

        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        # Oxygen decreases each step
        if self._step_counter % 3 == 0:
            self._oxygen -= 1
        if self._oxygen <= 0:
            self._on_life_lost()
            self._message = "Out of oxygen!"
            if not self._game_over:
                self._oxygen = 100
                self._player_x = self._WIDTH // 2
                self._player_y = self._SURFACE_Y + 3

        # Player movement
        if action_name == "UP" and self._player_y > self._SURFACE_Y:
            self._player_y -= 1
            self._player_dir = (0, -1)
        elif (
            action_name == "DOWN"
            and self._player_y < self._SEABED_Y - 1
        ):
            self._player_y += 1
            self._player_dir = (0, 1)
        elif action_name == "LEFT":
            self._facing = -1
            self._player_dir = (-1, 0)
            if self._player_x > 1:
                self._player_x -= 1
        elif action_name == "RIGHT":
            self._facing = 1
            self._player_dir = (1, 0)
            if self._player_x < self._WIDTH - 2:
                self._player_x += 1
        elif (
            action_name == "FIRE"
            and len(self._torpedoes) < 2
        ):
            tx = self._player_x + self._facing
            t = self._add_entity(
                "torpedo", "*", tx, self._player_y,
                dx=self._facing,
            )
            self._torpedoes.append(t)
        elif action_name == "SURFACE":
            if self._player_y <= self._SURFACE_Y + 1:
                self._oxygen = 100
                pts = self._carried * 50
                if self._carried > 0:
                    self._on_point_scored(pts)
                    reward += pts
                    self._rescued += self._carried
                    self._message = (
                        f"Surfaced! {self._carried} "
                        f"divers rescued! +{pts}"
                    )
                    self._carried = 0
                else:
                    self._message = "Oxygen refilled!"

        # Move torpedoes
        for t in self._torpedoes:
            if not t.alive:
                continue
            t.x += t.dx
            if t.x <= 0 or t.x >= self._WIDTH - 1:
                t.alive = False

        # Torpedo-enemy collisions
        for t in self._torpedoes:
            if not t.alive:
                continue
            for e in self._enemies:
                if (
                    e.alive
                    and e.x == t.x
                    and e.y == t.y
                ):
                    e.alive = False
                    t.alive = False
                    self._on_point_scored(10)
                    reward += 10
                    self._message = "Enemy hit! +10"
                    break
        self._torpedoes = [
            t for t in self._torpedoes if t.alive
        ]

        # Move enemies
        if self._step_counter % 2 == 0:
            for e in self._enemies:
                if not e.alive:
                    continue
                e.x += e.dx
                if e.x <= 1 or e.x >= self._WIDTH - 2:
                    e.alive = False

        # Enemy fire
        if self._step_counter % 8 == 0:
            shooters = [
                e for e in self._enemies
                if e.alive and e.data.get("shoots")
            ]
            if shooters and len(self._enemy_torpedoes) < 2:
                s = shooters[
                    int(self.rng.integers(len(shooters)))
                ]
                et = self._add_entity(
                    "enemy_torpedo", "·", s.x, s.y,
                    dx=-s.dx,
                )
                self._enemy_torpedoes.append(et)

        # Move enemy torpedoes
        for et in self._enemy_torpedoes:
            if not et.alive:
                continue
            et.x += et.dx
            if et.x <= 0 or et.x >= self._WIDTH - 1:
                et.alive = False
            elif (
                et.x == self._player_x
                and et.y == self._player_y
            ):
                et.alive = False
                self._on_life_lost()
                self._message = "Torpedo hit!"
                self._player_x = self._WIDTH // 2
                self._player_y = self._SURFACE_Y + 3
        self._enemy_torpedoes = [
            et for et in self._enemy_torpedoes if et.alive
        ]

        # Move divers
        if self._step_counter % 3 == 0:
            for d in self._divers:
                if not d.alive:
                    continue
                dd = d.data.get("dir", 1)
                d.x += dd
                if d.x <= 1 or d.x >= self._WIDTH - 2:
                    d.data["dir"] = -dd

        # Pick up divers
        for d in self._divers:
            if (
                d.alive
                and abs(d.x - self._player_x) <= 1
                and d.y == self._player_y
                and self._carried < 4
            ):
                d.alive = False
                self._carried += 1
                self._message = (
                    f"Diver picked up! ({self._carried}/4)"
                )

        # Enemy collision with player
        for e in self._enemies:
            if (
                e.alive
                and e.x == self._player_x
                and e.y == self._player_y
            ):
                e.alive = False
                self._on_life_lost()
                self._message = "Enemy collision!"
                self._player_x = self._WIDTH // 2
                self._player_y = self._SURFACE_Y + 3

        self._enemies = [e for e in self._enemies if e.alive]
        self._divers = [d for d in self._divers if d.alive]

        # Respawn enemies if all gone
        if not self._enemies:
            self._level += 1
            self._message = "Wave cleared!"
            # Respawn only enemies, keep divers/oxygen
            rng = self.rng
            n = min(3 + self._level, 8)
            for i in range(n):
                ey = int(
                    rng.integers(
                        self._SURFACE_Y + 1, self._SEABED_Y,
                    )
                )
                if rng.random() < 0.5:
                    ex, edx, ch = 1, 1, "→"
                else:
                    ex = self._WIDTH - 2
                    edx, ch = -1, "←"
                e = self._add_entity(
                    "enemy", ch, ex, ey, dx=edx,
                )
                e.data["shoots"] = i % 3 == 0
                self._enemies.append(e)

        self._redraw()
        info["oxygen"] = self._oxygen
        info["carried"] = self._carried
        info["rescued"] = self._rescued
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        render = [row[:] for row in self._grid]
        symbols: dict[str, str] = {}
        for y in range(self._grid_h):
            for x in range(self._grid_w):
                ch = render[y][x]
                if ch not in symbols:
                    symbols[ch] = self._symbol_meaning(ch)
        for e in self._entities:
            if (
                e.alive
                and 0 <= e.x < self._grid_w
                and 0 <= e.y < self._grid_h
            ):
                render[e.y][e.x] = e.char
                if e.char not in symbols:
                    symbols[e.char] = e.etype
        px, py = self._player_x, self._player_y
        if 0 <= px < self._grid_w and 0 <= py < self._grid_h:
            pch = self._DIR_CHARS.get(
                self._player_dir, "@"
            )
            render[py][px] = pch
            dname = self._DIR_NAMES.get(
                self._player_dir, "none"
            )
            symbols[pch] = f"you (facing {dname})"
        facing = "right" if self._facing == 1 else "left"
        hud = (
            f"Score: {self._score}    Lives: {self._lives}"
            f"    Level: {self._level}"
            f"    O2: {self._oxygen}%"
            f"    Divers: {self._carried}/4\n"
            f"Facing: {facing}"
        )
        return GridObservation(
            grid=grid_to_string(render),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _redraw(self) -> None:
        for y in range(2, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")
        # Water surface
        for x in range(1, self._WIDTH - 1):
            self._set_cell(x, 1, "~")
        # Enemies
        for e in self._enemies:
            if e.alive:
                self._set_cell(e.x, e.y, e.char)
        # Divers
        for d in self._divers:
            if d.alive:
                self._set_cell(d.x, d.y, "D")
        # Torpedoes
        for t in self._torpedoes:
            if t.alive:
                self._set_cell(t.x, t.y, "*")
        for et in self._enemy_torpedoes:
            if et.alive:
                self._set_cell(et.x, et.y, "·")

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "border", "│": "border",
            "~": "water surface", "=": "seabed",
            "→": "enemy (moving right)",
            "←": "enemy (moving left)",
            "D": "diver", "*": "your torpedo",
            "·": "enemy torpedo", " ": "water",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Submarine combat: shoot enemies and rescue divers. "
            "Pick up divers (max 4) then SURFACE to score. "
            "Watch your oxygen -- surface to refill."
        )
