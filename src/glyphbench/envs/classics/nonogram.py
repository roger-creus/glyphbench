"""Nonogram (Picross) puzzle: fill cells based on row/column clues."""

from __future__ import annotations

from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.base_env import BaseAsciiEnv
from glyphbench.core.observation import GridObservation
from glyphbench.core.registry import register_env

# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

_SYM_FILLED = "\u25a0"   # ■
_SYM_MARKED = "\u00d7"   # ×
_SYM_UNKNOWN = "?"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_clues(line: list[bool]) -> list[int]:
    """Compute the nonogram clue for a single row or column."""
    clues: list[int] = []
    count = 0
    for cell in line:
        if cell:
            count += 1
        else:
            if count > 0:
                clues.append(count)
            count = 0
    if count > 0:
        clues.append(count)
    return clues if clues else [0]


def _make_action_spec(grid_size: int) -> ActionSpec:
    """Create action spec for FILL_i and MARK_i actions."""
    n = grid_size * grid_size
    names: list[str] = []
    descs: list[str] = []
    for i in range(n):
        r, c = divmod(i, grid_size)
        names.append(f"FILL_{i}")
        descs.append(f"fill cell at row {r}, col {c}")
    for i in range(n):
        r, c = divmod(i, grid_size)
        names.append(f"MARK_{i}")
        descs.append(f"mark cell at row {r}, col {c} as empty")
    return ActionSpec(names=tuple(names), descriptions=tuple(descs))


# Pre-build action specs for both variants
_EASY_SIZE = 5
_HARD_SIZE = 8
_EASY_ACTION_SPEC = _make_action_spec(_EASY_SIZE)
_HARD_ACTION_SPEC = _make_action_spec(_HARD_SIZE)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class NonogramEnv(BaseAsciiEnv):
    """Nonogram puzzle. Fill/mark cells to match hidden pattern."""

    # Overridden per variant
    action_spec = _EASY_ACTION_SPEC
    noop_action_name = "FILL_0"  # re-filling same cell is harmless

    def __init__(self, grid_size: int = 5, max_turns: int = 200) -> None:
        self._grid_size = grid_size
        super().__init__(max_turns=max_turns)
        # Solution grid (True = filled)
        self._solution: np.ndarray = np.zeros((grid_size, grid_size), dtype=bool)
        # Player state: 0=unknown, 1=filled, 2=marked_empty
        self._player: np.ndarray = np.zeros((grid_size, grid_size), dtype=np.int8)
        # Clues
        self._row_clues: list[list[int]] = []
        self._col_clues: list[list[int]] = []
        self._completed: bool = False
        self._last_msg: str = ""

    def env_id(self) -> str:
        if self._grid_size == _EASY_SIZE:
            return "glyphbench/classics-nonogram-easy-v0"
        return "glyphbench/classics-nonogram-hard-v0"

    # ------------------------------------------------------------------
    # Puzzle generation
    # ------------------------------------------------------------------

    def _generate_puzzle(self) -> None:
        """Generate a random nonogram puzzle."""
        n = self._grid_size
        # Random pattern: ~50% fill rate
        self._solution = self.rng.random((n, n)) < 0.5

        # Ensure at least one filled cell per row and column
        for r in range(n):
            if not np.any(self._solution[r]):
                self._solution[r, int(self.rng.integers(n))] = True
        for c in range(n):
            if not np.any(self._solution[:, c]):
                self._solution[int(self.rng.integers(n)), c] = True

        # Compute clues
        self._row_clues = []
        for r in range(n):
            self._row_clues.append(_compute_clues(list(self._solution[r])))

        self._col_clues = []
        for c in range(n):
            self._col_clues.append(_compute_clues(list(self._solution[:, c])))

    # ------------------------------------------------------------------
    # Completion check
    # ------------------------------------------------------------------

    def _check_complete(self) -> bool:
        """Check if all cells are correctly resolved."""
        n = self._grid_size
        for r in range(n):
            for c in range(n):
                if self._solution[r, c]:
                    if self._player[r, c] != 1:
                        return False
                else:
                    if self._player[r, c] != 2:
                        return False
        return True

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        n = self._grid_size
        self._player = np.zeros((n, n), dtype=np.int8)
        self._completed = False
        self._last_msg = ""
        self._generate_puzzle()
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        n = self._grid_size
        n2 = n * n
        reward = 0.0

        if action < n2:
            # FILL action
            idx = action
            r, c = divmod(idx, n)
            if self._player[r, c] == 0:
                self._player[r, c] = 1
                if self._solution[r, c]:
                    reward = 0.01
                    self._last_msg = f"Filled ({r},{c}): correct!"
                else:
                    reward = -0.01
                    self._last_msg = f"Filled ({r},{c}): incorrect."
            else:
                self._last_msg = f"Cell ({r},{c}) already set."
        else:
            # MARK action
            idx = action - n2
            r, c = divmod(idx, n)
            if self._player[r, c] == 0:
                self._player[r, c] = 2
                if not self._solution[r, c]:
                    reward = 0.01
                    self._last_msg = f"Marked ({r},{c}) empty: correct!"
                else:
                    reward = -0.01
                    self._last_msg = f"Marked ({r},{c}) empty: incorrect."
            else:
                self._last_msg = f"Cell ({r},{c}) already set."

        # Check completion
        if self._check_complete():
            self._completed = True
            reward += 1.0
            self._last_msg = "Puzzle complete! You win!"
            info["win"] = True

        return self._render_current_observation(), reward, self._completed, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        n = self._grid_size
        grid = make_empty_grid(n, n, fill=_SYM_UNKNOWN)
        syms: dict[str, str] = {
            _SYM_UNKNOWN: "unknown cell",
            _SYM_FILLED: "filled",
            _SYM_MARKED: "marked empty",
        }

        state_to_sym = {0: _SYM_UNKNOWN, 1: _SYM_FILLED, 2: _SYM_MARKED}
        for r in range(n):
            for c in range(n):
                grid[r][c] = state_to_sym[int(self._player[r, c])]

        legend = build_legend(syms)

        # Build clue strings
        row_clue_strs: list[str] = []
        for r in range(n):
            row_clue_strs.append(f"  Row {r}: {' '.join(map(str, self._row_clues[r]))}")
        col_clue_strs: list[str] = []
        for c in range(n):
            col_clue_strs.append(f"  Col {c}: {' '.join(map(str, self._col_clues[c]))}")

        filled_count = int(np.sum(self._player > 0))
        total_cells = n * n

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Resolved: {filled_count} / {total_cells}\n"
            "Row clues:\n" + "\n".join(row_clue_strs) + "\n"
            "Column clues:\n" + "\n".join(col_clue_strs)
        )

        return GridObservation(
            grid=grid_to_string(grid), legend=legend, hud=hud, message=self._last_msg
        )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def system_prompt(self) -> str:
        n = self._grid_size
        return (
            f"You are playing {self.env_id()} -- Nonogram (Picross) puzzle.\n\n"
            "RULES\n"
            f"The grid is {n}x{n}. Each row and column has clues: sequences of "
            "numbers indicating consecutive runs of filled cells in that line.\n"
            "For example, clue '2 1' means there is a run of 2 filled cells, "
            "at least one gap, then a run of 1 filled cell.\n"
            "Clue '0' means no cells are filled in that line.\n\n"
            "ACTIONS\n"
            f"FILL_i (i=0..{n*n-1}): mark cell as filled (row=i//{n}, col=i%{n}).\n"
            f"MARK_i (i=0..{n*n-1}): mark cell as empty.\n"
            "Each cell can only be set once.\n\n"
            "SCORING\n"
            "+0.01 per correct action, -0.01 per incorrect.\n"
            "+1.0 bonus when puzzle is complete.\n\n"
            "STRATEGY\n"
            "Start with rows/columns that have large clues (most constrained). "
            "Use overlap logic: if a clue forces certain cells to be filled "
            "regardless of position, fill those first.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Variant classes
# ---------------------------------------------------------------------------


class NonogramEasyEnv(NonogramEnv):
    """5x5 Nonogram."""

    action_spec = _EASY_ACTION_SPEC
    noop_action_name = "FILL_0"

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(grid_size=_EASY_SIZE, max_turns=max_turns)


class NonogramHardEnv(NonogramEnv):
    """8x8 Nonogram."""

    action_spec = _HARD_ACTION_SPEC
    noop_action_name = "FILL_0"

    def __init__(self, max_turns: int = 300) -> None:
        super().__init__(grid_size=_HARD_SIZE, max_turns=max_turns)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_env(
    "glyphbench/classics-nonogram-easy-v0",
    "glyphbench.envs.classics.nonogram:NonogramEasyEnv",
)

register_env(
    "glyphbench/classics-nonogram-hard-v0",
    "glyphbench.envs.classics.nonogram:NonogramHardEnv",
)
