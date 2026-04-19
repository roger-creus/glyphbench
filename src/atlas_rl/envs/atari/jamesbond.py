"""Atari JamesBond environment.

Side-scrolling vehicle action. Dodge and destroy obstacles.

Gym ID: atlas_rl/atari-jamesbond-v0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas_rl.core.action import ActionSpec

from .base import AtariBase

_W = 30
_H = 16
_GROUND_Y = 13
_DIRS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


class JamesBondEnv(AtariBase):
    """JamesBond: side-scrolling vehicle action.

    Drive a vehicle, shoot enemies (E) and obstacles (X).
    Collect intel pickups (i) for bonus points.
    The world scrolls left; dodge or destroy threats.

    Grid: 30x16.
    Reward: +10 per obstacle, +20 per enemy, +50 per intel.
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE",
        ),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "move up",
            "move down",
            "fire weapon forward",
        ),
    )

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._scroll_timer: int = 0
        self._spawn_timer: int = 0
        self._distance: int = 0

    def env_id(self) -> str:
        return "atlas_rl/atari-jamesbond-v0"

    def _generate_level(self, seed: int) -> None:
        rng = np.random.default_rng(
            seed + self._level * 1117
        )
        self._init_grid(_W, _H)
        self._entities = []
        self._scroll_timer = 0
        self._spawn_timer = 0
        self._distance = 0

        # Top/bottom borders
        for x in range(_W):
            self._set_cell(x, 0, "-")
            self._set_cell(x, _H - 1, "-")

        # Ground
        for x in range(_W):
            self._set_cell(x, _GROUND_Y, "=")
            self._set_cell(x, _GROUND_Y + 1, "~")

        # Road markings
        for x in range(0, _W, 4):
            self._set_cell(x, _GROUND_Y - 3, ".")

        # Initial obstacles and enemies
        for _ in range(3 + self._level):
            ox = int(rng.integers(15, _W - 2))
            oy = int(rng.integers(3, _GROUND_Y - 1))
            if rng.random() < 0.4:
                self._add_entity("enemy", "E", ox, oy, dx=-1)
            else:
                self._add_entity(
                    "obstacle", "X", ox, oy, dx=-1
                )

        # Intel pickups
        for _ in range(2):
            ix = int(rng.integers(15, _W - 2))
            iy = int(rng.integers(3, _GROUND_Y - 1))
            self._add_entity("intel", "i", ix, iy, dx=-1)

        # Player position
        self._player_x = 3
        self._player_y = _GROUND_Y - 2

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        fire = action_name == "FIRE"

        # Move player
        if action_name in _DIRS:
            d = _DIRS[action_name]
            nx = self._player_x + d[0]
            ny = self._player_y + d[1]
            if 1 <= nx <= _W // 2 and 2 <= ny < _GROUND_Y:
                self._player_x = nx
                self._player_y = ny

        # Fire bullet
        if fire:
            self._add_entity(
                "bullet", ">",
                self._player_x + 1, self._player_y,
                dx=2,
            )

        # Move bullets
        for e in self._entities:
            if e.etype == "bullet" and e.alive:
                e.x += e.dx
                if e.x >= _W - 1:
                    e.alive = False

        # Scroll world: move enemies/obstacles left
        self._scroll_timer += 1
        if self._scroll_timer >= 2:
            self._scroll_timer = 0
            self._distance += 1
            for e in self._entities:
                if e.etype in ("enemy", "obstacle", "intel"):
                    if e.alive:
                        e.x += e.dx
                        if e.x < 1:
                            e.alive = False

        # Bullet-enemy/obstacle collisions
        for b in self._entities:
            if b.etype != "bullet" or not b.alive:
                continue
            for t in self._entities:
                if not t.alive:
                    continue
                if t.etype == "enemy" and b.x == t.x and b.y == t.y:
                    b.alive = False
                    t.alive = False
                    self._on_point_scored(20)
                    reward += 20
                    self._message = "Enemy destroyed! +20"
                elif (
                    t.etype == "obstacle"
                    and b.x == t.x
                    and b.y == t.y
                ):
                    b.alive = False
                    t.alive = False
                    self._on_point_scored(10)
                    reward += 10
                    self._message = "Obstacle destroyed! +10"

        # Player pickups
        for e in self._entities:
            if (
                e.etype == "intel"
                and e.alive
                and e.x == self._player_x
                and e.y == self._player_y
            ):
                e.alive = False
                self._on_point_scored(50)
                reward += 50
                self._message = "Intel collected! +50"

        # Player-enemy/obstacle collision
        for e in self._entities:
            if not e.alive:
                continue
            if e.etype in ("enemy", "obstacle"):
                if (
                    e.x == self._player_x
                    and e.y == self._player_y
                ):
                    self._on_life_lost()
                    self._message = f"Hit by {e.etype}!"
                    if not self._game_over:
                        self._player_x = 3
                        self._player_y = _GROUND_Y - 2
                    return reward, self._game_over, info

        self._entities = [
            e for e in self._entities if e.alive
        ]

        # Spawn new threats
        self._spawn_timer += 1
        rate = max(8 - self._level, 3)
        if self._spawn_timer >= rate:
            self._spawn_timer = 0
            oy = int(
                self.rng.integers(3, _GROUND_Y - 1)
            )
            r = self.rng.random()
            if r < 0.3:
                self._add_entity(
                    "enemy", "E", _W - 2, oy, dx=-1,
                )
            elif r < 0.7:
                self._add_entity(
                    "obstacle", "X", _W - 2, oy, dx=-1,
                )
            else:
                self._add_entity(
                    "intel", "i", _W - 2, oy, dx=-1,
                )

        # Level progression based on distance
        if self._distance >= 80 + self._level * 20:
            self._level += 1
            self._message = "Mission stage complete!"
            self._generate_level(self._level * 2999)

        # Redraw ground decorations
        for x in range(0, _W, 4):
            off = (x + self._distance) % 4
            self._set_cell(
                (x - off) % _W, _GROUND_Y - 3, "."
            )

        info["distance"] = self._distance
        return reward, self._game_over, info

    def _advance_entities(self) -> None:
        pass

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "-": "border",
            "=": "ground",
            "~": "water",
            ".": "road marking",
            "E": "enemy vehicle",
            "X": "obstacle",
            "i": "intel pickup",
            ">": "bullet",
            " ": "sky",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Drive your vehicle and survive. FIRE to shoot "
            "enemies (E) and obstacles (X). Collect intel (i) "
            "for bonus points. Use UP/DOWN/LEFT/RIGHT to dodge."
        )
