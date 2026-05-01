"""miniatari Q*bert.

Identity: Hop diagonally on cubes of a 4-row pyramid; paint each cube once.
Win condition: paint all 10 cubes of the pyramid.
Reward: Pattern A, +1/10 per fresh cube painted.
Loss: hop off the pyramid (terminates without -1).

Gym ID: glyphbench/miniatari-qbert-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=3, mean_return=+0.100
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniQbertEnv(MiniatariBase):
    """Mini Q*bert: 12x12 grid, 4-row pyramid (10 cubes total).

    Pyramid is laid out at logical coordinates (row, col) with row=0..3
    and col=0..row. Each cube maps to a grid cell using a fixed
    isometric-ish mapping: gx = 5 - row + 2*col, gy = 1 + 2*row.
    Cube starts unpainted ('.'); after the agent hops onto it the first
    time it becomes painted ('#') and grants +1/10. The agent (Q) starts
    on the apex (row=0, col=0). Each tick the agent picks one of 4
    diagonal directions: UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT.
    Falling off the pyramid (no destination cube) ends the run. Painting
    all 10 cubes wins.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT"),
        descriptions=(
            "do nothing",
            "hop diagonally up-left to the cube at (row-1, col-1)",
            "hop diagonally up-right to the cube at (row-1, col)",
            "hop diagonally down-left to the cube at (row+1, col)",
            "hop diagonally down-right to the cube at (row+1, col+1)",
        ),
    )

    default_max_turns = 300

    _WIDTH = 12
    _HEIGHT = 12
    _NUM_ROWS = 4
    _WIN_TARGET = 10  # 1+2+3+4 = 10

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._row: int = 0
        self._col: int = 0
        # painted[row][col] for col in 0..row
        self._painted: list[list[bool]] = []
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-qbert-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._row = 0
        self._col = 0
        self._painted = [[False] * (r + 1) for r in range(self._NUM_ROWS)]
        # Paint apex on entry
        self._painted[0][0] = True
        self._progress += 1
        self._sync_player_pos()

    def _grid_pos(self, row: int, col: int) -> tuple[int, int]:
        gx = 5 - row + 2 * col
        gy = 1 + 2 * row
        return gx, gy

    def _sync_player_pos(self) -> None:
        gx, gy = self._grid_pos(self._row, self._col)
        self._player_x = gx
        self._player_y = gy

    def _valid(self, row: int, col: int) -> bool:
        return 0 <= row < self._NUM_ROWS and 0 <= col <= row

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        new_row, new_col = self._row, self._col
        if action_name == "UP_LEFT":
            new_row, new_col = self._row - 1, self._col - 1
        elif action_name == "UP_RIGHT":
            new_row, new_col = self._row - 1, self._col
        elif action_name == "DOWN_LEFT":
            new_row, new_col = self._row + 1, self._col
        elif action_name == "DOWN_RIGHT":
            new_row, new_col = self._row + 1, self._col + 1

        if action_name != "NOOP":
            if not self._valid(new_row, new_col):
                self._message = "Hopped off the pyramid!"
                self._game_over = True
                self._won = False
                return reward, True, info
            self._row, self._col = new_row, new_col
            self._sync_player_pos()
            if not self._painted[self._row][self._col]:
                self._painted[self._row][self._col] = True
                self._progress += 1
                reward += self._progress_reward(self._WIN_TARGET)
                self._message = f"Cube painted! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info

        info["progress"] = self._progress
        info["row"] = self._row
        info["col"] = self._col
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        for r in range(self._NUM_ROWS):
            for c in range(r + 1):
                gx, gy = self._grid_pos(r, c)
                if 0 <= gx < self._WIDTH and 0 <= gy < self._HEIGHT:
                    grid[gy][gx] = "#" if self._painted[r][c] else "."
        # Player
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Q"

        symbols = {
            " ": "void",
            ".": "unpainted cube",
            "#": "painted cube",
            "Q": "you (Q*bert)",
        }
        unpainted = sum(
            1 for r in range(self._NUM_ROWS) for c in range(r + 1) if not self._painted[r][c]
        )
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Painted: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}\n"
            f"Position: row={self._row}, col={self._col}    "
            f"Unpainted: {unpainted}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Q*bert on a 12x12 grid. A 4-row pyramid has 10 cubes "
            "(. unpainted, # painted) at logical (row, col) positions with "
            "row=0..3 and col=0..row. You (Q) start on the apex which is "
            "auto-painted. Each tick choose one diagonal hop: UP_LEFT, "
            "UP_RIGHT, DOWN_LEFT, DOWN_RIGHT. Hopping onto an unpainted "
            "cube paints it and grants +1/10. Hopping off the pyramid "
            "(invalid destination) ends the run. Paint all 10 cubes to win. "
            "Reward: +1/10 per fresh cube painted."
        )
