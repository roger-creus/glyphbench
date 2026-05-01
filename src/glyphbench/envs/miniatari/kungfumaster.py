"""miniatari Kung-Fu Master.

Identity: Side-scrolling brawl; punch enemies that approach.
Win condition: defeat all 8 enemies.
Reward: Pattern D, +1/8 per enemy KO; -1 if an enemy reaches you.

Gym ID: glyphbench/miniatari-kungfumaster-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=21, mean_return=-0.871
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniKungFuMasterEnv(MiniatariBase):
    """Mini Kung-Fu Master: 16x8 corridor, 8 enemies attack from both sides.

    The corridor is 16x8 with floor at row 5 and ceiling row 0. Player
    (Y, arrow facing) starts in the center. Enemies (e) spawn alternately
    from left/right edges every 4 ticks (up to 8 total spawned). Enemies
    walk toward the player at 1 cell every 2 ticks. PUNCH knocks out any
    enemy in either cell directly adjacent to you (left or right, range
    1). KICK knocks out any enemy within 2 cells on either side (range
    2). The strikes hit both sides because there is no separate "face"
    action and enemies attack from both edges. Spawned enemies are
    tracked; +1/8 per KO. If any enemy ends a turn on your cell, -1
    terminal.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "PUNCH", "KICK"),
        descriptions=(
            "do nothing",
            "step left and face left",
            "step right and face right",
            "punch the cells adjacent on both sides (range 1 each side)",
            "kick the cells within range 2 on both sides",
        ),
    )

    default_max_turns = 300

    _WIDTH = 16
    _HEIGHT = 8
    _N_ENEMIES = 8
    _WIN_TARGET = _N_ENEMIES
    _FLOOR_Y = 5
    _ENEMY_MOVE_EVERY = 2
    _ENEMY_SPAWN_EVERY = 4

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._enemies: list[list[int]] = []  # [x, dir]; y is fixed at FLOOR_Y
        self._spawned: int = 0
        self._kos: int = 0
        self._tick_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-kungfumaster-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._tick_count = 0
        self._spawned = 0
        self._kos = 0
        self._enemies = []
        self._player_x = self._WIDTH // 2
        self._player_y = self._FLOOR_Y
        self._player_dir = (1, 0)

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Player move
        if action_name == "LEFT" and self._player_x > 0:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._player_x < self._WIDTH - 1:
            self._player_x += 1
            self._player_dir = (1, 0)

        # 2. Combat — PUNCH/KICK hit BOTH sides (not just facing direction).
        # Without a face-only action there is no way for the agent to attack
        # an enemy on the side opposite their facing without first moving
        # toward them. Instead we let PUNCH cover the cells immediately to
        # the left and right of the player; KICK extends the reach to 2.
        kos_this_step = 0
        if action_name in ("PUNCH", "KICK"):
            reach = 1 if action_name == "PUNCH" else 2
            survivors: list[list[int]] = []
            for ex, edir in self._enemies:
                hit = False
                for k in range(1, reach + 1):
                    if ex == self._player_x + k or ex == self._player_x - k:
                        hit = True
                        break
                if hit:
                    self._kos += 1
                    kos_this_step += 1
                else:
                    survivors.append([ex, edir])
            self._enemies = survivors
            for _ in range(kos_this_step):
                reward += self._progress_reward(self._WIN_TARGET)
            if kos_this_step:
                self._message = f"KO! ({self._kos}/{self._WIN_TARGET})"
                if self._kos >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info

        # 3. Enemies move toward player
        if self._tick_count % self._ENEMY_MOVE_EVERY == 0:
            for e in self._enemies:
                if e[0] < self._player_x:
                    e[0] += 1
                    e[1] = 1
                elif e[0] > self._player_x:
                    e[0] -= 1
                    e[1] = -1

        # 4. Spawn alternate sides
        if self._spawned < self._N_ENEMIES and self._tick_count % self._ENEMY_SPAWN_EVERY == 0:
            side = -1 if self._spawned % 2 == 0 else 1
            spawn_x = 0 if side == -1 else self._WIDTH - 1
            self._enemies.append([spawn_x, -side])
            self._spawned += 1

        # 5. Enemy reaches player?
        for ex, _edir in self._enemies:
            if ex == self._player_x:
                self._message = "Enemy got you!"
                reward += self._death_reward()
                self._on_life_lost()
                return reward, True, info

        info["spawned"] = self._spawned
        info["kos"] = self._kos
        info["progress"] = self._kos
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Floor row
        for x in range(self._WIDTH):
            grid[self._FLOOR_Y + 1][x] = "─"
        # Enemies
        for ex, _edir in self._enemies:
            if 0 <= ex < self._WIDTH:
                grid[self._FLOOR_Y][ex] = "e"
        # Player
        if 0 <= self._player_x < self._WIDTH:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._FLOOR_Y][self._player_x] = pch

        symbols = {
            " ": "open air",
            "─": "floor",
            "e": "enemy",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"you (facing {self._DIR_NAMES.get(self._player_dir, 'right')})"

        enemies_info = " ".join(f"x={ex}" for ex, _ in self._enemies)
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"KOs: {self._kos}/{self._WIN_TARGET}    "
            f"Spawned: {self._spawned}/{self._N_ENEMIES}    "
            f"Score: {self._score:.3f}\n"
            f"You x={self._player_x}    Enemies: {enemies_info}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Kung-Fu Master on a 16x8 corridor. You (Y, arrow shows "
            "facing) stand on row 5. Enemies (e) spawn alternately from "
            "left/right every 4 ticks (up to 8 total) and walk toward you "
            "1 cell every 2 ticks. LEFT/RIGHT moves and faces. PUNCH KOs "
            "an enemy in either cell immediately adjacent to you (left "
            "OR right, range 1 per side). KICK KOs any enemy within 2 "
            "cells on either side (range 2 per side). Both strikes hit "
            "both sides because there is no face-only action. Reward: "
            "+1/8 per KO. If any enemy ends a tick on your cell, -1 "
            "terminal."
        )
