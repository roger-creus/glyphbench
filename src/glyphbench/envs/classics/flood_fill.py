"""Flood Fill -- absorb the board by changing your region's color.

Gym IDs:
  glyphbench/classics-floodfill-easy-v0   (8x8, 25 moves)
  glyphbench/classics-floodfill-hard-v0   (12x12, 35 moves)
"""

from __future__ import annotations

from collections import deque
from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.base_env import BaseAsciiEnv
from glyphbench.core.observation import GridObservation
from glyphbench.core.registry import register_env

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_COLORS = 6
COLOR_SYMBOLS = ("\u25cf", "\u25cb", "\u25c6", "\u25c7", "\u25a0", "\u25a1")
# ● ○ ◆ ◇ ■ □
COLOR_NAMES = ("red", "blue", "green", "yellow", "purple", "orange")

FLOOD_ACTION_SPEC = ActionSpec(
    names=("COLOR_0", "COLOR_1", "COLOR_2", "COLOR_3", "COLOR_4", "COLOR_5"),
    descriptions=(
        f"change region to {COLOR_NAMES[0]} ({COLOR_SYMBOLS[0]})",
        f"change region to {COLOR_NAMES[1]} ({COLOR_SYMBOLS[1]})",
        f"change region to {COLOR_NAMES[2]} ({COLOR_SYMBOLS[2]})",
        f"change region to {COLOR_NAMES[3]} ({COLOR_SYMBOLS[3]})",
        f"change region to {COLOR_NAMES[4]} ({COLOR_SYMBOLS[4]})",
        f"change region to {COLOR_NAMES[5]} ({COLOR_SYMBOLS[5]})",
    ),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flood_region(board: list[list[int]], rows: int, cols: int) -> set[tuple[int, int]]:
    """BFS from (0,0) to find the connected region of the same color."""
    color = board[0][0]
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    queue.append((0, 0))
    visited.add((0, 0))
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if board[nr][nc] == color:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return visited


def _apply_flood(board: list[list[int]], rows: int, cols: int, new_color: int) -> int:
    """Change the connected region from (0,0) to new_color. Return cells absorbed."""
    old_color = board[0][0]
    if old_color == new_color:
        return 0
    region = _flood_region(board, rows, cols)
    # Find cells adjacent to region with new_color (they'll be absorbed)
    old_region_size = len(region)

    # Set all region cells to new color
    for r, c in region:
        board[r][c] = new_color

    # Count new region size after the color change
    new_region = _flood_region(board, rows, cols)
    absorbed = len(new_region) - old_region_size
    return absorbed


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

class _FloodFillBase(BaseAsciiEnv):
    """Change your region's color to absorb adjacent same-color cells."""

    action_spec = FLOOD_ACTION_SPEC
    noop_action_name: str = "COLOR_0"

    _grid_rows: int = 8
    _grid_cols: int = 8
    _max_steps: int = 25
    _difficulty: str = "easy"

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns if max_turns is not None else self._max_steps)
        self._board: list[list[int]] = []
        self._total_cells: int = 0
        self._region_size: int = 0
        self._filled: bool = False

    def env_id(self) -> str:
        return f"glyphbench/classics-floodfill-{self._difficulty}-v0"

    def _reset(self, seed: int) -> GridObservation:
        self._total_cells = self._grid_rows * self._grid_cols
        # Generate random board
        self._board = [
            [int(self.rng.integers(0, NUM_COLORS)) for _ in range(self._grid_cols)]
            for _ in range(self._grid_rows)
        ]
        region = _flood_region(self._board, self._grid_rows, self._grid_cols)
        self._region_size = len(region)
        self._filled = False
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        new_color = action  # action index = color index

        old_color = self._board[0][0]
        if new_color == old_color:
            # Choosing current color is a NOOP -- wasted move
            info["wasted_move"] = True
            return self._render_current_observation(), 0.0, False, False, info

        absorbed = _apply_flood(self._board, self._grid_rows, self._grid_cols, new_color)
        self._region_size += absorbed
        reward = 0.01 * absorbed

        # Check if filled
        if self._region_size >= self._total_cells:
            self._filled = True
            reward += 1.0
            return self._render_current_observation(), reward, True, False, info

        info["region_size"] = self._region_size
        return self._render_current_observation(), reward, False, False, info

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(self._grid_cols, self._grid_rows, " ")

        for r in range(self._grid_rows):
            for c in range(self._grid_cols):
                grid[r][c] = COLOR_SYMBOLS[self._board[r][c]]

        legend_map: dict[str, str] = {}
        for i in range(NUM_COLORS):
            legend_map[COLOR_SYMBOLS[i]] = f"color {i} ({COLOR_NAMES[i]})"

        pct = 100.0 * self._region_size / self._total_cells
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Region: {self._region_size} / {self._total_cells} ({pct:.0f}%)    "
            f"Current color: {COLOR_SYMBOLS[self._board[0][0]]} ({COLOR_NAMES[self._board[0][0]]})"
        )

        msg = ""
        if self._filled:
            msg = "Board filled! You win!"

        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(legend_map),
            hud=hud,
            message=msg,
        )

    def system_prompt(self) -> str:
        sym_list = " ".join(f"{COLOR_SYMBOLS[i]}={COLOR_NAMES[i]}" for i in range(NUM_COLORS))
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Fill the entire board with one color by repeatedly changing your region's color.\n\n"
            "RULES\n"
            f"- The board is {self._grid_rows}x{self._grid_cols} with {NUM_COLORS} colors: {sym_list}.\n"
            "- Your region starts from the top-left corner (0,0).\n"
            "- Each turn, choose a NEW color. Your entire connected region changes to that color, "
            "absorbing any adjacent cells already of that color.\n"
            "- Choosing your current color wastes a move.\n"
            f"- You have {self._max_steps} moves to fill the entire board.\n"
            "- +0.01 reward per cell absorbed, +1 bonus for filling the whole board.\n"
            "- Strategy: pick the color that borders the most cells of your current region.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------

class FloodFillEasyEnv(_FloodFillBase):
    _grid_rows = 8
    _grid_cols = 8
    _max_steps = 25
    _difficulty = "easy"


class FloodFillHardEnv(_FloodFillBase):
    _grid_rows = 12
    _grid_cols = 12
    _max_steps = 35
    _difficulty = "hard"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_env(
    "glyphbench/classics-floodfill-easy-v0",
    "glyphbench.envs.classics.flood_fill:FloodFillEasyEnv",
)
register_env(
    "glyphbench/classics-floodfill-hard-v0",
    "glyphbench.envs.classics.flood_fill:FloodFillHardEnv",
)
