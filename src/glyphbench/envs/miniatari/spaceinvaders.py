"""miniatari Space Invaders.

Identity: Bottom cannon defends against descending invaders.
Win condition: clear all 8 invaders in the wave.
Reward: Pattern D, +1/8 per kill; -1 if invaders reach the cannon row.
Loss: invaders reach bottom (terminal -1).

Gym ID: glyphbench/miniatari-spaceinvaders-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=51, mean_return=-0.546
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniSpaceInvadersEnv(MiniatariBase):
    """Mini Space Invaders: 14x12 grid, single 8-invader wave.

    Cannon at the bottom row (y=11). Invaders form a 2x4 block
    starting at top-left, sweeping right→left→down. Bullet travels
    instantly up the column when FIRE is pressed (first invader in
    column dies). Invaders advance one row every K ticks. Game ends
    when invaders reach the cannon row (-1) or all are destroyed (+1).
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move cannon left one cell",
            "move cannon right one cell",
            "fire a bullet straight up your column",
        ),
    )

    default_max_turns = 200

    _WIDTH = 14
    _HEIGHT = 12
    _WIN_TARGET = 8
    _CANNON_Y = 11
    _INVADER_ROWS = 2
    _INVADER_COLS = 4
    _ADVANCE_EVERY = 8  # ticks per invader-row drop
    _SWEEP_EVERY = 2    # ticks per invader-column shift

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._invaders: set[tuple[int, int]] = set()
        self._sweep_dir: int = 1
        self._wave_x: int = 0
        self._wave_y: int = 0
        self._tick_count: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-spaceinvaders-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        self._sweep_dir = 1
        self._wave_x = 1
        self._wave_y = 1
        # 2x4 grid of invaders, spaced 2 cells apart horizontally
        self._invaders = set()
        for r in range(self._INVADER_ROWS):
            for c in range(self._INVADER_COLS):
                self._invaders.add((c * 2, r))
        self._player_x = self._WIDTH // 2
        self._player_y = self._CANNON_Y

    def _occupied(self, x: int, y: int) -> bool:
        """Check if any invader sits at the absolute (x, y) cell."""
        for ix, iy in self._invaders:
            ax = self._wave_x + ix
            ay = self._wave_y + iy
            if ax == x and ay == y:
                return True
        return False

    def _abs_invader_cells(self) -> set[tuple[int, int]]:
        return {(self._wave_x + ix, self._wave_y + iy) for ix, iy in self._invaders}

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Cannon move
        if action_name == "LEFT" and self._player_x > 0:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._player_x < self._WIDTH - 1:
            self._player_x += 1
            self._player_dir = (1, 0)

        # 2. Fire bullet (instantaneous column-shot)
        if action_name == "FIRE":
            # Find lowest invader in player's column
            col_x = self._player_x
            target: tuple[int, int] | None = None
            for ix, iy in self._invaders:
                ax = self._wave_x + ix
                ay = self._wave_y + iy
                if ax == col_x and 0 <= ay < self._CANNON_Y:
                    if target is None or ay > self._wave_y + target[1]:
                        target = (ix, iy)
            if target is not None:
                self._invaders.discard(target)
                reward += self._progress_reward(self._WIN_TARGET)
                self._progress += 1
                self._message = f"Hit! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info

        # 3. Wave sweeps every _SWEEP_EVERY ticks
        if self._tick_count % self._SWEEP_EVERY == 0 and self._invaders:
            xs = [self._wave_x + ix for ix, _iy in self._invaders]
            min_x, max_x = min(xs), max(xs)
            new_wave_x = self._wave_x + self._sweep_dir
            new_min = min_x + self._sweep_dir
            new_max = max_x + self._sweep_dir
            if new_min < 0 or new_max >= self._WIDTH:
                # Reverse and drop
                self._sweep_dir *= -1
                self._wave_y += 1
            else:
                self._wave_x = new_wave_x

        # 4. Wave drops every _ADVANCE_EVERY ticks (regardless of bounce)
        if self._tick_count % self._ADVANCE_EVERY == 0 and self._invaders:
            self._wave_y += 1

        # 5. Reach cannon row?
        if self._invaders:
            ys = [self._wave_y + iy for _ix, iy in self._invaders]
            max_y = max(ys)
            if max_y >= self._CANNON_Y:
                self._message = "Invaders reached the cannon row!"
                reward = self._death_reward()
                self._on_life_lost()
                return reward, True, info

        info["progress"] = self._progress
        info["invaders_left"] = len(self._invaders)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Cannon row line
        for x in range(self._WIDTH):
            grid[self._CANNON_Y][x] = "_"
        # Invaders
        for ix, iy in self._invaders:
            ax, ay = self._wave_x + ix, self._wave_y + iy
            if 0 <= ax < self._WIDTH and 0 <= ay < self._HEIGHT:
                grid[ay][ax] = "M"
        # Cannon
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Y"

        symbols = {
            " ": "empty space",
            "_": "cannon row (defense line)",
            "M": "invader",
            "Y": "your cannon",
        }

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Killed: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}\n"
            f"Cannon x={self._player_x}    "
            f"Wave shift=({self._wave_x},{self._wave_y}) sweep_dir={self._sweep_dir:+d}"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Space Invaders on a 14x12 grid. A 2x4 wave of 8 invaders "
            "(M) sweeps left/right, dropping one row when it reaches an "
            "edge AND every 8 ticks regardless. Your cannon (Y) sits on "
            "the bottom row. LEFT/RIGHT slides the cannon. FIRE destroys "
            "the lowest invader in your current column instantly. Clear "
            "all 8 invaders to win. If any invader reaches the cannon row "
            "you take a -1 terminal penalty. Reward: +1/8 per kill, -1 if "
            "the wave breaks through."
        )
