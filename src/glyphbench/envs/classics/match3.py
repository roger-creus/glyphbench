"""Match-3 gem-swapping puzzle game."""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.ascii_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.base_env import BaseAsciiEnv
from glyphbench.core.observation import GridObservation
from glyphbench.core.registry import register_env

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SIZE = 8
_NUM_GEMS = 6

_GEM_SYMS = ("\u2666", "\u2663", "\u2660", "\u2665", "\u2605", "\u25cf")
# ♦ ♣ ♠ ♥ ★ ●
_GEM_NAMES = ("diamond", "club", "spade", "heart", "star", "circle")
_SYM_EMPTY = "\u00b7"  # ·

# Directions for swap
_DIR_OFFSETS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # UP, DOWN, LEFT, RIGHT
_DIR_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

# ---------------------------------------------------------------------------
# Action spec: 256 actions = 8*8*4
# ---------------------------------------------------------------------------

_action_names: list[str] = []
_action_descs: list[str] = []
for _r in range(_SIZE):
    for _c in range(_SIZE):
        for _d in range(4):
            _action_names.append(f"SWAP_{_r}_{_c}_{_DIR_NAMES[_d]}")
            _action_descs.append(
                f"swap gem at ({_r},{_c}) with the one {_DIR_NAMES[_d].lower()}"
            )

MATCH3_ACTION_SPEC = ActionSpec(
    names=tuple(_action_names), descriptions=tuple(_action_descs)
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class Match3Env(BaseAsciiEnv):
    """Match-3 gem-swapping on an 8x8 board with 6 gem types."""

    action_spec = MATCH3_ACTION_SPEC
    noop_action_name = "SWAP_0_0_UP"  # invalid swap acts as noop

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        # Board stores gem indices 0.._NUM_GEMS-1, or -1 for empty
        self._board: list[list[int]] = []
        self._score: int = 0
        self._total_matched: int = 0
        self._no_moves: bool = False
        self._last_msg: str = ""

    def env_id(self) -> str:
        return "glyphbench/classics-match3-v0"

    # ------------------------------------------------------------------
    # Board helpers
    # ------------------------------------------------------------------

    def _fill_board(self) -> None:
        """Fill the board ensuring no initial matches."""
        for r in range(_SIZE):
            for c in range(_SIZE):
                while True:
                    gem = int(self.rng.integers(_NUM_GEMS))
                    self._board[r][c] = gem
                    # Check horizontal match
                    if c >= 2 and self._board[r][c - 1] == gem and self._board[r][c - 2] == gem:
                        continue
                    # Check vertical match
                    if r >= 2 and self._board[r - 1][c] == gem and self._board[r - 2][c] == gem:
                        continue
                    break

    def _find_matches(self) -> set[tuple[int, int]]:
        """Find all cells involved in matches of 3+."""
        matched: set[tuple[int, int]] = set()

        # Horizontal
        for r in range(_SIZE):
            c = 0
            while c < _SIZE:
                gem = self._board[r][c]
                if gem < 0:
                    c += 1
                    continue
                run_len = 1
                while c + run_len < _SIZE and self._board[r][c + run_len] == gem:
                    run_len += 1
                if run_len >= 3:
                    for k in range(run_len):
                        matched.add((r, c + k))
                c += run_len

        # Vertical
        for c in range(_SIZE):
            r = 0
            while r < _SIZE:
                gem = self._board[r][c]
                if gem < 0:
                    r += 1
                    continue
                run_len = 1
                while r + run_len < _SIZE and self._board[r + run_len][c] == gem:
                    run_len += 1
                if run_len >= 3:
                    for k in range(run_len):
                        matched.add((r + k, c))
                r += run_len

        return matched

    def _remove_and_score(self, matched: set[tuple[int, int]]) -> int:
        """Remove matched cells. Returns count of matched gems (1 pt per gem).

        Simple linear scoring decouples strategy from lucky cascades: a smart
        player who sets up a 5-match outscores a random swap that happens to
        trigger a chain.
        """
        count = len(matched)
        for r, c in matched:
            self._board[r][c] = -1
        return count

    def _gravity(self) -> None:
        """Drop gems down to fill empty spaces."""
        for c in range(_SIZE):
            write_row = _SIZE - 1
            for r in range(_SIZE - 1, -1, -1):
                if self._board[r][c] >= 0:
                    self._board[write_row][c] = self._board[r][c]
                    if write_row != r:
                        self._board[r][c] = -1
                    write_row -= 1
            # Fill top with new gems
            for r in range(write_row, -1, -1):
                self._board[r][c] = int(self.rng.integers(_NUM_GEMS))

    def _cascade(self) -> int:
        """Process all matches + gravity cascades. Returns the score for
        ONLY the initial (player-triggered) match. Subsequent cascades from
        falling gems still happen mechanically but contribute 0 reward.

        This decouples strategy from luck: random swaps can no longer farm
        points via accidental long chains.
        """
        # First wave: score == count of gems in the player's triggered match.
        chain = 0
        first_wave_score = 0
        while True:
            matched = self._find_matches()
            if not matched:
                break
            chain += 1
            score = self._remove_and_score(matched)
            self._total_matched += len(matched)
            if chain == 1:
                first_wave_score = score
            # cascades (chain >= 2) score 0
            self._gravity()
        return first_wave_score

    def _has_valid_move(self) -> bool:
        """Check if any swap creates a match."""
        for r in range(_SIZE):
            for c in range(_SIZE):
                for d in range(4):
                    dr, dc = _DIR_OFFSETS[d]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < _SIZE and 0 <= nc < _SIZE:
                        # Swap
                        self._board[r][c], self._board[nr][nc] = self._board[nr][nc], self._board[r][c]
                        has_match = len(self._find_matches()) > 0
                        # Swap back
                        self._board[r][c], self._board[nr][nc] = self._board[nr][nc], self._board[r][c]
                        if has_match:
                            return True
        return False

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._board = [[-1] * _SIZE for _ in range(_SIZE)]
        self._score = 0
        self._total_matched = 0
        self._no_moves = False
        self._last_msg = ""
        self._fill_board()
        # Ensure at least one valid move
        if not self._has_valid_move():
            self._fill_board()  # retry
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}

        # Decode action
        remainder = action
        r = remainder // (_SIZE * 4)
        remainder = remainder % (_SIZE * 4)
        c = remainder // 4
        d = remainder % 4

        dr, dc = _DIR_OFFSETS[d]
        nr, nc = r + dr, c + dc

        # Validate swap
        if not (0 <= nr < _SIZE and 0 <= nc < _SIZE):
            self._last_msg = "Invalid swap (out of bounds). No effect."
            return self._render_current_observation(), 0.0, False, False, info

        # Perform swap
        self._board[r][c], self._board[nr][nc] = self._board[nr][nc], self._board[r][c]

        # Check if swap creates a match
        matched = self._find_matches()
        if not matched:
            # Swap back -- invalid move
            self._board[r][c], self._board[nr][nc] = self._board[nr][nc], self._board[r][c]
            self._last_msg = "Swap did not create a match. No effect."
            return self._render_current_observation(), 0.0, False, False, info

        # Process cascades
        score = self._cascade()
        self._score += score
        reward = float(score)

        self._last_msg = f"Match! +{score} points."

        # Check for game over (no valid moves)
        if not self._has_valid_move():
            self._no_moves = True
            info["no_moves"] = True
            self._last_msg += " No more valid moves. Game over."
            return self._render_current_observation(), reward, True, False, info

        return self._render_current_observation(), reward, False, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(_SIZE, _SIZE, fill=_SYM_EMPTY)
        syms: dict[str, str] = {}

        for r in range(_SIZE):
            for c in range(_SIZE):
                gem = self._board[r][c]
                if gem >= 0:
                    sym = _GEM_SYMS[gem]
                    grid[r][c] = sym
                    syms[sym] = _GEM_NAMES[gem]
                else:
                    syms[_SYM_EMPTY] = "empty (during cascade)"

        legend = build_legend(syms)
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Score: {self._score}    "
            f"Total matched: {self._total_matched}"
        )

        return GridObservation(
            grid=grid_to_string(grid), legend=legend, hud=hud, message=self._last_msg
        )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()} -- a Match-3 puzzle game.\n\n"
            "RULES\n"
            f"The board is {_SIZE}x{_SIZE} with {_NUM_GEMS} gem types.\n"
            "Each turn you swap two adjacent gems (horizontal or vertical).\n"
            "The swap only takes effect if it creates a match of 3+ in a row/column.\n"
            "After matches, gems above fall down and new gems fill from the top.\n"
            "Cascades from falling gems continue the turn but do NOT multiply score.\n\n"
            "SCORING\n"
            "  1 point per gem matched. 3-match = 3 points, 4-match = 4, 5-match = 5.\n"
            "  Cascade matches add their gem counts (no multiplier).\n\n"
            "ACTIONS\n"
            f"Actions are named SWAP_<row>_<col>_<DIRECTION> with row,col in [0,{_SIZE - 1}] "
            "and DIRECTION in {UP, DOWN, LEFT, RIGHT}. Swaps out of bounds are NOOP.\n"
            f"Total action space: {_SIZE * _SIZE * 4} actions.\n"
            "Examples:\n"
            "  SWAP_0_0_RIGHT — swap gem at (0,0) with the one on its right\n"
            "  SWAP_3_4_DOWN  — swap gem at (3,4) with the one below it\n"
            f"  SWAP_{_SIZE - 1}_{_SIZE - 1}_UP — swap gem at bottom-right with the one above\n"
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_env(
    "glyphbench/classics-match3-v0",
    "glyphbench.envs.classics.match3:Match3Env",
)
