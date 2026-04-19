"""Atari RoadRunner environment.

Chase platformer. Collect birdseed, avoid Coyote.

Gym ID: atlas_rl/atari-roadrunner-v0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas_rl.core.action import ActionSpec

from .base import AtariBase

_W = 30
_H = 16
_GROUND_Y = 13
_ROAD_Y = 11


class RoadRunnerEnv(AtariBase):
    """RoadRunner: collect seeds, outrun the Coyote.

    The road scrolls left. Collect birdseed (o) and avoid
    the Coyote (W) who chases you. Jump over obstacles.
    Mines (*) can stun the Coyote if he hits them.

    Grid: 30x16.
    Reward: +10 per seed, +50 if Coyote hits mine.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "JUMP"),
        descriptions=(
            "do nothing",
            "run left",
            "run right",
            "jump over obstacle or gap",
        ),
    )

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._scroll_timer: int = 0
        self._distance: int = 0
        self._jumping: bool = False
        self._jump_timer: int = 0
        self._coyote_stun: int = 0
        self._spawn_timer: int = 0

    def env_id(self) -> str:
        return "atlas_rl/atari-roadrunner-v0"

    def _generate_level(self, seed: int) -> None:
        rng = np.random.default_rng(
            seed + self._level * 1231
        )
        self._init_grid(_W, _H)
        self._entities = []
        self._scroll_timer = 0
        self._distance = 0
        self._jumping = False
        self._jump_timer = 0
        self._coyote_stun = 0
        self._spawn_timer = 0

        # Draw landscape
        for x in range(_W):
            self._set_cell(x, 0, "-")
            self._set_cell(x, _H - 1, "-")
            self._set_cell(x, _GROUND_Y, "=")
            # Desert floor below
            for y in range(_GROUND_Y + 1, _H - 1):
                self._set_cell(x, y, ".")

        # Road surface
        for x in range(_W):
            self._set_cell(x, _ROAD_Y, "_")
            self._set_cell(x, _ROAD_Y + 1, "_")

        # Initial seeds on road
        for i in range(5):
            sx = int(rng.integers(10, _W - 2))
            self._add_entity("seed", "o", sx, _ROAD_Y)

        # Initial obstacles (rocks)
        for _ in range(2):
            ox = int(rng.integers(15, _W - 2))
            self._add_entity(
                "rock", "#", ox, _ROAD_Y, dx=-1,
            )

        # Mines
        for _ in range(2):
            mx = int(rng.integers(12, _W - 2))
            self._add_entity("mine", "*", mx, _ROAD_Y)

        # Coyote (chases from behind)
        self._add_entity(
            "coyote", "W", 1, _ROAD_Y, dx=0,
        )

        # Player (Road Runner) on the road
        self._player_x = 8
        self._player_y = _ROAD_Y

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Player movement
        if action_name == "LEFT":
            if self._player_x > 1:
                self._player_x -= 1
        elif action_name == "RIGHT":
            if self._player_x < _W - 2:
                self._player_x += 1
        elif action_name == "JUMP":
            if not self._jumping:
                self._jumping = True
                self._jump_timer = 4
                self._player_y = _ROAD_Y - 2

        # Jump physics
        if self._jumping:
            self._jump_timer -= 1
            if self._jump_timer <= 0:
                self._jumping = False
                self._player_y = _ROAD_Y

        # Scroll world
        self._scroll_timer += 1
        if self._scroll_timer >= 2:
            self._scroll_timer = 0
            self._distance += 1
            for e in self._entities:
                if e.etype in ("seed", "rock", "mine"):
                    if e.alive:
                        e.x -= 1
                        if e.x < 1:
                            e.alive = False

        # Coyote AI
        coyote = None
        for e in self._entities:
            if e.etype == "coyote" and e.alive:
                coyote = e
                break

        if coyote and coyote.alive:
            if self._coyote_stun > 0:
                self._coyote_stun -= 1
            else:
                # Chase player
                speed = 1 if self.rng.random() < 0.6 else 0
                if coyote.x < self._player_x:
                    coyote.x += speed
                elif coyote.x > self._player_x:
                    coyote.x -= speed

        # Seed collection
        for e in self._entities:
            if (
                e.etype == "seed"
                and e.alive
                and e.x == self._player_x
                and abs(e.y - self._player_y) <= 1
            ):
                e.alive = False
                self._on_point_scored(10)
                reward += 10
                self._message = "Birdseed! +10"

        # Rock collision
        for e in self._entities:
            if (
                e.etype == "rock"
                and e.alive
                and e.x == self._player_x
                and e.y == self._player_y
            ):
                self._on_life_lost()
                self._message = "Hit a rock!"
                if not self._game_over:
                    self._player_x = 8
                    self._player_y = _ROAD_Y
                return reward, self._game_over, info

        # Coyote-mine collision
        if coyote and coyote.alive:
            for e in self._entities:
                if (
                    e.etype == "mine"
                    and e.alive
                    and e.x == coyote.x
                    and e.y == coyote.y
                ):
                    e.alive = False
                    self._coyote_stun = 20
                    coyote.x = max(1, coyote.x - 8)
                    self._on_point_scored(50)
                    reward += 50
                    self._message = "Coyote hit mine! +50"

        # Coyote catches player
        if (
            coyote
            and coyote.alive
            and self._coyote_stun <= 0
            and coyote.x == self._player_x
            and abs(coyote.y - self._player_y) <= 1
        ):
            self._on_life_lost()
            self._message = "Caught by Coyote!"
            if not self._game_over:
                self._player_x = 8
                self._player_y = _ROAD_Y
                coyote.x = 1
            return reward, self._game_over, info

        self._entities = [
            e for e in self._entities if e.alive
        ]

        # Spawn new items
        self._spawn_timer += 1
        if self._spawn_timer >= max(6 - self._level, 3):
            self._spawn_timer = 0
            r = self.rng.random()
            if r < 0.4:
                self._add_entity(
                    "seed", "o", _W - 2, _ROAD_Y,
                )
            elif r < 0.7:
                self._add_entity(
                    "rock", "#", _W - 2, _ROAD_Y, dx=-1,
                )
            else:
                self._add_entity(
                    "mine", "*", _W - 2, _ROAD_Y,
                )

        # Level progression
        if self._distance >= 100 + self._level * 30:
            self._level += 1
            self._message = "Stage complete!"
            self._generate_level(self._level * 4007)

        info["distance"] = self._distance
        return reward, self._game_over, info

    def _advance_entities(self) -> None:
        pass

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "-": "border",
            "=": "ground",
            "_": "road",
            ".": "desert",
            "o": "birdseed",
            "#": "rock obstacle",
            "*": "mine",
            "W": "Coyote",
            " ": "sky",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Run along the road as the Road Runner. "
            "Collect birdseed (o), avoid rocks (#), and "
            "outrun the Coyote (W). JUMP over obstacles. "
            "Lure Coyote into mines (*) for bonus points."
        )
