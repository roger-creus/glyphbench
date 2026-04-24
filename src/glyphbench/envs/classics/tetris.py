"""Classic Tetris falling blocks game."""

from __future__ import annotations

from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WIDTH = 10
_HEIGHT = 20

_SYM_EMPTY = "\u00b7"  # ·
_SYM_ACTIVE = "\u2593"  # ▓

# Locked piece symbols per type
_PIECE_SYMS = {
    "I": "\u25b0",  # ▰
    "O": "\u25ae",  # ▮
    "T": "\u25b2",  # ▲
    "S": "\u25c6",  # ◆
    "Z": "\u25c7",  # ◇
    "J": "\u25c4",  # ◄
    "L": "\u25ba",  # ►
}

# Tetromino shapes: list of (row, col) offsets. Row 0 = top.
# Each piece has 4 rotations.
_SHAPES: dict[str, list[list[tuple[int, int]]]] = {
    "I": [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
    ],
    "O": [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    "T": [
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 0), (1, 0), (2, 0), (1, 1)],
        [(1, 0), (1, 1), (1, 2), (0, 1)],
        [(0, 0), (1, 0), (2, 0), (1, -1)],
    ],
    "S": [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
    "Z": [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],
    "J": [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, -1)],
    ],
    "L": [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
}

_PIECE_TYPES = list(_SHAPES.keys())

# ---------------------------------------------------------------------------
# Action spec
# ---------------------------------------------------------------------------

TETRIS_ACTION_SPEC = ActionSpec(
    names=("LEFT", "RIGHT", "ROTATE_CW", "ROTATE_CCW", "HARD_DROP", "NOOP"),
    descriptions=(
        "move piece one cell left",
        "move piece one cell right",
        "rotate piece 90 degrees clockwise",
        "rotate piece 90 degrees counter-clockwise",
        "instantly drop piece to the lowest valid position",
        "do nothing (piece drops 1 row by gravity)",
    ),
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TetrisEnv(BaseGlyphEnv):
    """Classic Tetris on a 10x20 board."""

    action_spec = TETRIS_ACTION_SPEC
    noop_action_name = "NOOP"

    def __init__(self, max_turns: int = 2000) -> None:
        super().__init__(max_turns=max_turns)
        # Board: stores piece type char or empty string
        self._board: list[list[str]] = []
        # Current piece state
        self._piece_type: str = "I"
        self._piece_rot: int = 0
        self._piece_row: int = 0
        self._piece_col: int = 0
        self._game_over: bool = False
        self._score: int = 0
        self._lines_cleared: int = 0

    def env_id(self) -> str:
        return "glyphbench/classics-tetris-v0"

    # ------------------------------------------------------------------
    # Piece helpers
    # ------------------------------------------------------------------

    def _get_cells(self, ptype: str, rot: int, row: int, col: int) -> list[tuple[int, int]]:
        """Return list of (row, col) for the piece at given position."""
        return [(row + dr, col + dc) for dr, dc in _SHAPES[ptype][rot % 4]]

    def _valid_position(self, ptype: str, rot: int, row: int, col: int) -> bool:
        for r, c in self._get_cells(ptype, rot, row, col):
            if r < 0 or r >= _HEIGHT or c < 0 or c >= _WIDTH:
                return False
            if self._board[r][c] != "":
                return False
        return True

    def _spawn_piece(self) -> bool:
        """Spawn a new random piece at the top. Returns False if blocked."""
        self._piece_type = _PIECE_TYPES[int(self.rng.integers(len(_PIECE_TYPES)))]
        self._piece_rot = 0
        self._piece_row = 0
        self._piece_col = _WIDTH // 2 - 1
        return self._valid_position(
            self._piece_type, self._piece_rot, self._piece_row, self._piece_col
        )

    def _lock_piece(self) -> None:
        """Lock the current piece into the board."""
        for r, c in self._get_cells(
            self._piece_type, self._piece_rot, self._piece_row, self._piece_col
        ):
            if 0 <= r < _HEIGHT and 0 <= c < _WIDTH:
                self._board[r][c] = self._piece_type

    def _clear_lines(self) -> int:
        """Clear full lines and return number cleared."""
        cleared = 0
        new_board: list[list[str]] = []
        for r in range(_HEIGHT):
            if all(self._board[r][c] != "" for c in range(_WIDTH)):
                cleared += 1
            else:
                new_board.append(self._board[r])
        # Add empty rows at top
        while len(new_board) < _HEIGHT:
            new_board.insert(0, [""] * _WIDTH)
        self._board = new_board
        return cleared

    def _drop_one(self) -> bool:
        """Try to drop piece one row. Returns True if successful."""
        if self._valid_position(
            self._piece_type, self._piece_rot, self._piece_row + 1, self._piece_col
        ):
            self._piece_row += 1
            return True
        return False

    def _hard_drop(self) -> None:
        """Drop piece as far as it goes."""
        while self._drop_one():
            pass

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._board = [[""] * _WIDTH for _ in range(_HEIGHT)]
        self._game_over = False
        self._score = 0
        self._lines_cleared = 0
        if not self._spawn_piece():
            self._game_over = True
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        reward = 0.0

        if self._game_over:
            return self._render_current_observation(), 0.0, True, False, info

        name = self.action_spec.names[action]

        # Apply action
        if name == "LEFT":
            if self._valid_position(
                self._piece_type, self._piece_rot, self._piece_row, self._piece_col - 1
            ):
                self._piece_col -= 1
        elif name == "RIGHT":
            if self._valid_position(
                self._piece_type, self._piece_rot, self._piece_row, self._piece_col + 1
            ):
                self._piece_col += 1
        elif name == "ROTATE_CW":
            new_rot = (self._piece_rot + 1) % 4
            if self._valid_position(
                self._piece_type, new_rot, self._piece_row, self._piece_col
            ):
                self._piece_rot = new_rot
        elif name == "ROTATE_CCW":
            new_rot = (self._piece_rot - 1) % 4
            if self._valid_position(
                self._piece_type, new_rot, self._piece_row, self._piece_col
            ):
                self._piece_rot = new_rot
        elif name == "HARD_DROP":
            self._hard_drop()
            # Lock immediately after hard drop
            self._lock_piece()
            cleared = self._clear_lines()
            if cleared > 0:
                self._lines_cleared += cleared
                reward_map = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}
                reward = reward_map.get(cleared, 8.0)
                self._score += int(reward)
            # Spawn next
            if not self._spawn_piece():
                self._game_over = True
                info["game_over"] = True
            return self._render_current_observation(), reward, self._game_over, False, info
        # NOOP: just gravity

        # Gravity: drop one row
        if not self._drop_one():
            # Can't drop -> lock
            self._lock_piece()
            cleared = self._clear_lines()
            if cleared > 0:
                self._lines_cleared += cleared
                reward_map = {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}
                reward = reward_map.get(cleared, 8.0)
                self._score += int(reward)
            # Spawn next
            if not self._spawn_piece():
                self._game_over = True
                info["game_over"] = True

        return self._render_current_observation(), reward, self._game_over, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(_WIDTH, _HEIGHT, fill=_SYM_EMPTY)
        syms: dict[str, str] = {_SYM_EMPTY: "empty"}

        # Draw locked pieces
        for r in range(_HEIGHT):
            for c in range(_WIDTH):
                pt = self._board[r][c]
                if pt != "":
                    sym = _PIECE_SYMS[pt]
                    grid[r][c] = sym
                    if sym not in syms:
                        syms[sym] = f"locked {pt}-piece"

        # Draw active piece
        if not self._game_over:
            for r, c in self._get_cells(
                self._piece_type, self._piece_rot, self._piece_row, self._piece_col
            ):
                if 0 <= r < _HEIGHT and 0 <= c < _WIDTH:
                    grid[r][c] = _SYM_ACTIVE
            syms[_SYM_ACTIVE] = "falling piece"

        legend = build_legend(syms)
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Score: {self._score}    "
            f"Lines: {self._lines_cleared}    "
            f"Piece: {self._piece_type}"
        )
        msg = "Game over! The board is full." if self._game_over else ""

        return GridObservation(
            grid=grid_to_string(grid), legend=legend, hud=hud, message=msg
        )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()} -- classic Tetris.\n\n"
            "RULES\n"
            f"The board is {_WIDTH} wide x {_HEIGHT} tall.\n"
            "Tetrominoes (I, O, T, S, Z, J, L) fall from the top.\n"
            "Each turn: your action is applied, then gravity drops the piece 1 row.\n"
            "If the piece cannot drop further, it locks in place and a new piece spawns.\n"
            "Complete horizontal lines are cleared for points:\n"
            "  1 line = +1, 2 lines = +3, 3 lines = +5, 4 lines (Tetris) = +8.\n"
            "The game ends when a new piece cannot spawn.\n\n"
            "STRATEGY\n"
            "Stack pieces flat. Keep the board low. Save space for line clears.\n"
            "Use HARD_DROP to instantly place a piece at the bottom.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

