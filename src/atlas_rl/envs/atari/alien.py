"""Atari Alien environment.

Action maze game. Kill aliens, collect eggs for points.

Gym ID: atlas_rl/atari-alien-v0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas_rl.core.action import ActionSpec
from atlas_rl.core.observation import GridObservation

from .base import AtariBase

_W = 20
_H = 20
_DIRS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


class AlienEnv(AtariBase):
    """Alien: maze action game.

    Navigate a maze, collect eggs (o), and shoot aliens (A).
    Aliens chase the player through corridors.
    Pulsar (P) spawns periodically and is invincible.

    Grid: 20x20.
    Reward: +10 per egg, +50 per alien killed.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move up",
            "move down",
            "move left",
            "move right",
            "shoot in last-faced direction",
        ),
    )

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._facing: tuple[int, int] = (1, 0)
        self._eggs_left: int = 0

    def env_id(self) -> str:
        return "atlas_rl/atari-alien-v0"

    def _generate_level(self, seed: int) -> None:
        rng = np.random.default_rng(seed + self._level * 997)
        self._init_grid(_W, _H)
        self._entities = []
        self._facing = (1, 0)

        # Border walls
        for x in range(_W):
            self._set_cell(x, 0, "#")
            self._set_cell(x, _H - 1, "#")
        for y in range(_H):
            self._set_cell(0, y, "#")
            self._set_cell(_W - 1, y, "#")

        # Maze: create corridors by carving from a filled grid
        for y in range(1, _H - 1):
            for x in range(1, _W - 1):
                self._set_cell(x, y, "#")

        # Carve corridors using simple grid pattern
        for y in range(2, _H - 2, 2):
            for x in range(2, _W - 2):
                self._set_cell(x, y, " ")
        for x in range(2, _W - 2, 4):
            for y in range(2, _H - 2):
                self._set_cell(x, y, " ")

        # Extra random openings
        n_extra = int(rng.integers(8, 16))
        for _ in range(n_extra):
            rx = int(rng.integers(2, _W - 2))
            ry = int(rng.integers(2, _H - 2))
            self._set_cell(rx, ry, " ")

        # Place eggs on open cells
        self._eggs_left = 0
        for y in range(2, _H - 2, 2):
            for x in range(2, _W - 2, 3):
                if self._grid_at(x, y) == " ":
                    if rng.random() < 0.4:
                        self._add_entity("egg", "o", x, y)
                        self._eggs_left += 1

        # Place aliens
        n_aliens = min(3 + self._level, 8)
        for _ in range(n_aliens):
            for _att in range(30):
                ax = int(rng.integers(3, _W - 3))
                ay = int(rng.integers(3, _H - 3))
                if self._grid_at(ax, ay) == " ":
                    a = self._add_entity("alien", "A", ax, ay)
                    a.data["timer"] = int(rng.integers(2, 6))
                    break

        # Player start
        self._player_x = 2
        self._player_y = 2
        self._set_cell(2, 2, " ")

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        fire = action_name == "FIRE"

        if action_name in _DIRS:
            d = _DIRS[action_name]
            self._facing = d
            self._player_dir = d
            nx = self._player_x + d[0]
            ny = self._player_y + d[1]
            if not self._is_solid(nx, ny):
                self._player_x = nx
                self._player_y = ny

        # Fire bullet
        if fire:
            bx = self._player_x + self._facing[0]
            by = self._player_y + self._facing[1]
            if not self._is_solid(bx, by):
                b = self._add_entity("bullet", "!", bx, by)
                b.dx = self._facing[0]
                b.dy = self._facing[1]
                b.data["owner"] = "player"

        # Move bullets
        for e in self._entities:
            if e.etype != "bullet" or not e.alive:
                continue
            e.x += e.dx
            e.y += e.dy
            if self._is_solid(e.x, e.y):
                e.alive = False

        # Alien AI: chase player
        for e in self._entities:
            if e.etype != "alien" or not e.alive:
                continue
            e.data["timer"] = e.data.get("timer", 3) - 1
            if e.data["timer"] > 0:
                continue
            e.data["timer"] = int(self.rng.integers(2, 5))
            dx = (
                1 if self._player_x > e.x
                else (-1 if self._player_x < e.x else 0)
            )
            dy = (
                1 if self._player_y > e.y
                else (-1 if self._player_y < e.y else 0)
            )
            if self.rng.random() < 0.5:
                dy = 0
            else:
                dx = 0
            nx, ny = e.x + dx, e.y + dy
            if not self._is_solid(nx, ny):
                e.x, e.y = nx, ny

        # Bullet-alien collisions
        for b in self._entities:
            if b.etype != "bullet" or not b.alive:
                continue
            for a in self._entities:
                if a.etype != "alien" or not a.alive:
                    continue
                if b.x == a.x and b.y == a.y:
                    b.alive = False
                    a.alive = False
                    self._on_point_scored(50)
                    reward += 50
                    self._message = "Alien killed! +50"

        # Player-egg collection
        for e in self._entities:
            if (
                e.etype == "egg"
                and e.alive
                and e.x == self._player_x
                and e.y == self._player_y
            ):
                e.alive = False
                self._on_point_scored(10)
                reward += 10
                self._eggs_left -= 1
                self._message = "Egg collected! +10"

        # Player-alien collision
        for e in self._entities:
            if (
                e.etype == "alien"
                and e.alive
                and e.x == self._player_x
                and e.y == self._player_y
            ):
                self._on_life_lost()
                self._message = "Caught by alien!"
                if not self._game_over:
                    self._player_x = 2
                    self._player_y = 2
                break

        self._entities = [e for e in self._entities if e.alive]

        # Level clear
        if self._eggs_left <= 0:
            self._level += 1
            self._message = "Level cleared!"
            self._generate_level(self._level * 7919)

        info["eggs_left"] = self._eggs_left
        return reward, self._game_over, info

    def _advance_entities(self) -> None:
        pass

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "#": "wall",
            " ": "corridor",
            "o": "egg",
            "A": "alien",
            "!": "bullet",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        dname = self._DIR_NAMES.get(self._facing, "none")
        extra = (
            f"Facing: {dname}  "
            f"Eggs: {self._eggs_left}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Navigate the maze, collect all eggs (o) and shoot "
            "aliens (A) with FIRE. Avoid alien contact."
        )
