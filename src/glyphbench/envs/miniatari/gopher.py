"""miniatari Gopher.

Identity: Defend a row of carrots from a digging gopher by filling holes.
Win condition: 3 carrots survive K turns.
Reward: Pattern D, +1/3 per surviving carrot at K turns; -1 if all 3 lost early.

Gym ID: glyphbench/miniatari-gopher-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=60, mean_return=+0.400
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniGopherEnv(MiniatariBase):
    """Mini Gopher: 14x8 grid; defend 3 carrots through 80 turns.

    Carrots (▼) sit at row 2 in columns 3, 7, 11. Below each carrot
    is a 3-cell column of dirt (▓). The gopher (G) tunnels at row 6
    and pops up to dig the dirt above whichever column it's under
    every 6 ticks. The player (Y, with shovel ↓) walks along row 1
    (just above the carrots) and FILL packs dirt into the column
    directly below them, restoring 1 dirt cell. If the gopher reaches
    row 2 (steals a carrot), that carrot is lost. After 80 ticks, +1/3
    per surviving carrot. Lose all 3 -> -1 terminal.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FILL"),
        descriptions=(
            "do nothing",
            "shuffle left along the surface",
            "shuffle right along the surface",
            "shovel one dirt cell back into the column below you",
        ),
    )

    default_max_turns = 300

    _WIDTH = 14
    _HEIGHT = 8
    _N_CARROTS = 3
    _WIN_TARGET = _N_CARROTS
    _CARROT_COLS = (3, 7, 11)
    _CARROT_Y = 2
    _DIRT_TOP = 3
    _DIRT_BOTTOM = 5  # rows 3, 4, 5 are dirt by default
    _GOPHER_Y = 6
    _PLAYER_Y = 1
    _GOPHER_DIG_EVERY = 6  # gopher digs every K ticks
    _GOPHER_MOVE_EVERY = 2  # gopher repositions every K ticks
    _DEFENSE_TURNS = 60

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        # dirt[col_idx] = top dirt row (CARROT_Y+1 down through DIRT_BOTTOM).
        # If a column has no dirt left and the gopher reaches CARROT_Y, the
        # carrot at that col is taken.
        self._dirt: dict[int, int] = {}
        self._carrots_alive: dict[int, bool] = {}
        self._gopher_x: int = 0
        self._tick_count: int = 0
        self._progress: int = 0  # carrots taken so far (informational)

    def env_id(self) -> str:
        return "glyphbench/miniatari-gopher-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._tick_count = 0
        self._progress = 0
        # Each carrot column: top of dirt block is row CARROT_Y+1 (=3).
        # Empty layer represented by top=DIRT_BOTTOM+1 means no dirt.
        self._dirt = {col: self._DIRT_TOP for col in self._CARROT_COLS}
        self._carrots_alive = {col: True for col in self._CARROT_COLS}
        self._player_x = self._CARROT_COLS[1]
        self._player_y = self._PLAYER_Y
        self._player_dir = (0, 1)
        # Gopher starts at the leftmost carrot column on its tunnel row
        rng = self.rng
        self._gopher_x = self._CARROT_COLS[int(rng.integers(0, self._N_CARROTS))]

    def _surviving_carrots(self) -> int:
        return sum(1 for v in self._carrots_alive.values() if v)

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

        # 2. FILL: pack dirt into player's column if it's a carrot column
        if action_name == "FILL":
            if self._player_x in self._dirt:
                cur_top = self._dirt[self._player_x]
                if cur_top > self._DIRT_TOP:
                    # Restore one dirt row at top
                    self._dirt[self._player_x] = cur_top - 1
                    self._message = "Packed dirt back!"

        # 3. Gopher repositions toward nearest alive carrot column
        if self._tick_count % self._GOPHER_MOVE_EVERY == 0:
            alive_cols = [c for c, alive in self._carrots_alive.items() if alive]
            if alive_cols:
                target = min(alive_cols, key=lambda c: abs(c - self._gopher_x))
                if self._gopher_x < target:
                    self._gopher_x += 1
                elif self._gopher_x > target:
                    self._gopher_x -= 1

        # 4. Gopher digs upward at its current column every DIG_EVERY ticks
        if self._tick_count % self._GOPHER_DIG_EVERY == 0:
            if (self._gopher_x in self._dirt and
                    self._carrots_alive.get(self._gopher_x, False)):
                cur_top = self._dirt[self._gopher_x]
                if cur_top <= self._DIRT_BOTTOM:
                    # Remove top dirt row
                    self._dirt[self._gopher_x] = cur_top + 1
                else:
                    # No dirt left -> gopher steals carrot
                    self._carrots_alive[self._gopher_x] = False
                    self._progress += 1
                    self._message = f"Gopher took a carrot! ({self._surviving_carrots()} left)"
                    if self._surviving_carrots() == 0:
                        reward = self._death_reward()
                        self._on_life_lost()
                        return reward, True, info

        # 5. End-of-defense check at DEFENSE_TURNS
        if self._tick_count >= self._DEFENSE_TURNS:
            survivors = self._surviving_carrots()
            reward = survivors * self._progress_reward(self._WIN_TARGET)
            self._message = f"Time's up — {survivors}/{self._WIN_TARGET} carrots safe."
            if survivors > 0:
                self._on_won()
            else:
                self._on_life_lost()
            return reward, True, info

        info["progress"] = self._progress
        info["carrots_left"] = self._surviving_carrots()
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Surface line (player walks here at row 1; ground row at row 2 below carrots)
        for x in range(self._WIDTH):
            grid[self._GOPHER_Y + 1][x] = "─"
        # Carrots
        for col in self._CARROT_COLS:
            if self._carrots_alive[col]:
                grid[self._CARROT_Y][col] = "▼"
        # Dirt (rows DIRT_TOP..DIRT_BOTTOM, only if dirt[col] <= row)
        for col in self._CARROT_COLS:
            top = self._dirt[col]
            for y in range(top, self._DIRT_BOTTOM + 1):
                grid[y][col] = "▓"
        # Gopher
        if 0 <= self._gopher_x < self._WIDTH:
            grid[self._GOPHER_Y][self._gopher_x] = "G"
        # Player
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Y"

        symbols = {
            " ": "open air",
            "─": "ground line",
            "▼": "carrot",
            "▓": "dirt",
            "G": "gopher",
            "Y": "you (with shovel)",
        }

        carrots_left = self._surviving_carrots()
        dirt_info = " ".join(
            f"col{c}:{self._DIRT_BOTTOM - self._dirt[c] + 1}" for c in self._CARROT_COLS
        )
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Carrots: {carrots_left}/{self._WIN_TARGET}    "
            f"Defense: {self._tick_count}/{self._DEFENSE_TURNS}    "
            f"Score: {self._score:.3f}\n"
            f"You x={self._player_x}    Gopher x={self._gopher_x}    "
            f"Dirt remaining: {dirt_info}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Gopher on a 14x8 field. Carrots (▼) sit at row 2 in "
            "columns 3, 7, 11; below each is a 3-cell dirt column (▓). "
            "The gopher (G) tunnels at row 6 and shifts toward the nearest "
            "alive carrot 1 cell every 2 ticks. Every 6 ticks the gopher "
            "removes 1 dirt row above its current column; once dirt is "
            "gone in that column, the carrot is stolen. You (Y) walk row "
            "1; LEFT/RIGHT moves you. FILL adds 1 dirt row back into the "
            "carrot column directly below you (clamped to 3 max). After "
            "60 ticks the episode ends. Reward: +1/3 per surviving "
            "carrot. If all 3 are stolen before time runs out, -1 "
            "terminal."
        )
