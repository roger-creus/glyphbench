"""2048 sliding tile game.

Gym IDs:
  glyphbench/classics-2048-v0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIZE = 4

SLIDE_ACTION_SPEC = ActionSpec(
    names=("UP", "DOWN", "LEFT", "RIGHT"),
    descriptions=(
        "slide all tiles up",
        "slide all tiles down",
        "slide all tiles left",
        "slide all tiles right",
    ),
)

SYM_EMPTY = "\u00b7"  # ·

# Tile display: values -> symbol string
# For values < 1024 use the number itself; 1024 -> K, 2048 -> W
_TILE_SYMS: dict[int, str] = {
    0: SYM_EMPTY,
    2: "2",
    4: "4",
    8: "8",
    16: "16",
    32: "32",
    64: "64",
    128: "128",
    256: "256",
    512: "512",
    1024: "K",
    2048: "W",
    4096: "4K",
    8192: "8K",
}


def _tile_sym(val: int) -> str:
    if val == 0:
        return SYM_EMPTY
    if val in _TILE_SYMS:
        return _TILE_SYMS[val]
    if val >= 2048:
        return "W"
    if val >= 1024:
        return "K"
    return str(val)


# ---------------------------------------------------------------------------
# 2048 Env
# ---------------------------------------------------------------------------


class Slide2048Env(BaseGlyphEnv):
    """2048: slide tiles to merge and reach high values."""

    action_spec = SLIDE_ACTION_SPEC
    noop_action_name: str = "UP"

    def __init__(self, max_turns: int = 2000) -> None:
        super().__init__(max_turns=max_turns)
        self._board: np.ndarray = np.zeros((SIZE, SIZE), dtype=np.int32)
        self._score: int = 0
        self._max_tile: int = 0

    def env_id(self) -> str:
        return "glyphbench/classics-2048-v0"

    # ------------------------------------------------------------------
    # Board helpers
    # ------------------------------------------------------------------

    def _spawn_tile(self) -> None:
        """Spawn a new 2 or 4 tile on a random empty cell."""
        empty = list(zip(*np.where(self._board == 0)))
        if not empty:
            return
        idx = int(self.rng.integers(0, len(empty)))
        r, c = empty[idx]
        self._board[r][c] = 2 if self.rng.random() < 0.9 else 4

    @staticmethod
    def _slide_row_left(row: np.ndarray) -> tuple[np.ndarray, int]:
        """Slide a single row left and merge. Returns (new_row, merge_score)."""
        # Remove zeros
        tiles = row[row != 0].tolist()
        merged: list[int] = []
        score = 0
        skip = False
        for i in range(len(tiles)):
            if skip:
                skip = False
                continue
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                val = tiles[i] * 2
                merged.append(val)
                score += val
                skip = True
            else:
                merged.append(tiles[i])
        # Pad with zeros
        merged.extend([0] * (SIZE - len(merged)))
        return np.array(merged, dtype=np.int32), score

    def _slide(self, direction: str) -> tuple[bool, int]:
        """Slide the board in the given direction. Returns (changed, merge_score)."""
        board = self._board.copy()
        total_score = 0

        if direction == "LEFT":
            for r in range(SIZE):
                self._board[r], s = self._slide_row_left(self._board[r])
                total_score += s
        elif direction == "RIGHT":
            for r in range(SIZE):
                self._board[r] = self._board[r][::-1]
                self._board[r], s = self._slide_row_left(self._board[r])
                self._board[r] = self._board[r][::-1]
                total_score += s
        elif direction == "UP":
            for c in range(SIZE):
                col = self._board[:, c].copy()
                self._board[:, c], s = self._slide_row_left(col)
                total_score += s
        elif direction == "DOWN":
            for c in range(SIZE):
                col = self._board[:, c][::-1].copy()
                new_col, s = self._slide_row_left(col)
                self._board[:, c] = new_col[::-1]
                total_score += s

        changed = not np.array_equal(board, self._board)
        return changed, total_score

    def _has_valid_moves(self) -> bool:
        """Check if any move can change the board."""
        if np.any(self._board == 0):
            return True
        # Check horizontal merges
        for r in range(SIZE):
            for c in range(SIZE - 1):
                if self._board[r][c] == self._board[r][c + 1]:
                    return True
        # Check vertical merges
        for r in range(SIZE - 1):
            for c in range(SIZE):
                if self._board[r][c] == self._board[r + 1][c]:
                    return True
        return False

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._board = np.zeros((SIZE, SIZE), dtype=np.int32)
        self._score = 0
        self._max_tile = 0
        self._spawn_tile()
        self._spawn_tile()
        self._max_tile = int(np.max(self._board))
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        name = self.action_spec.names[action]

        changed, merge_score = self._slide(name)

        if changed:
            self._spawn_tile()

        self._score += merge_score
        self._max_tile = int(np.max(self._board))
        reward = float(merge_score)

        terminated = not self._has_valid_moves()

        info["score"] = self._score
        info["max_tile"] = self._max_tile
        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        # Render as a padded grid for alignment
        cell_width = 4
        rows: list[str] = []
        for r in range(SIZE):
            cells: list[str] = []
            for c in range(SIZE):
                sym = _tile_sym(int(self._board[r][c]))
                cells.append(sym.rjust(cell_width))
            rows.append("".join(cells))
        grid_str = "\n".join(rows)

        # Collect unique tile symbols for legend
        sym_meanings: dict[str, str] = {SYM_EMPTY: "empty cell"}
        for r in range(SIZE):
            for c in range(SIZE):
                val = int(self._board[r][c])
                if val > 0:
                    sym = _tile_sym(val)
                    if sym not in sym_meanings:
                        sym_meanings[sym] = f"tile with value {val}"

        legend = build_legend(sym_meanings)

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Score: {self._score}    "
            f"Max tile: {self._max_tile}"
        )

        return GridObservation(grid=grid_str, legend=legend, hud=hud, message="")

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Slide tiles on a 4x4 grid to merge matching numbers and achieve the "
            "highest possible score.\n\n"
            "RULES\n"
            "- Tiles slide in the chosen direction until they hit a wall or another tile.\n"
            "- Two tiles with the same value merge into one with double the value.\n"
            "- After each valid move, a new tile (2 or 4) spawns on a random empty cell.\n"
            "- If no valid moves remain, the game ends.\n"
            "- Reward per step equals the sum of merged tile values.\n"
            "- Tiles: numbers show the value; K = 1024, W = 2048.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

