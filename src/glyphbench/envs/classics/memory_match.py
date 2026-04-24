"""Memory Match -- card-pair matching game.

Gym IDs:
  glyphbench/classics-memorymatch-easy-v0   (4x4 = 8 pairs)
  glyphbench/classics-memorymatch-hard-v0   (6x6 = 18 pairs)
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.base_env import BaseAsciiEnv
from glyphbench.core.observation import GridObservation
from glyphbench.core.registry import register_env

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYM_FACE_DOWN = "\u2592"  # ▒

# Letters for pairs: A=pair1, B=pair2, ...
PAIR_LETTERS = tuple("ABCDEFGHIJKLMNOPQR")  # up to 18 pairs

# ---------------------------------------------------------------------------
# Build action spec
# ---------------------------------------------------------------------------

def _build_action_spec(total_cells: int) -> ActionSpec:
    names: list[str] = []
    descs: list[str] = []
    for i in range(total_cells):
        names.append(f"FLIP_{i}")
        descs.append(f"flip card at position {i}")
    return ActionSpec(names=tuple(names), descriptions=tuple(descs))


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

class _MemoryMatchBase(BaseAsciiEnv):
    """Flip two cards per turn to find matching pairs."""

    noop_action_name: str = "FLIP_0"

    _rows: int = 4
    _cols: int = 4
    _difficulty: str = "easy"

    def __init__(self, max_turns: int = 100) -> None:
        self._total_cells = self._rows * self._cols
        self._num_pairs = self._total_cells // 2
        self.action_spec = _build_action_spec(self._total_cells)
        super().__init__(max_turns=max_turns)
        # Card values (0..num_pairs-1 for pair identity)
        self._board: list[int] = []
        self._matched: list[bool] = []  # True if card is matched (face up permanently)
        self._first_flip: int | None = None  # index of first flipped card this turn
        self._second_flip: int | None = None
        self._on_first_flip: bool = True  # True = waiting for first flip, False = waiting for second
        self._pairs_found: int = 0
        self._total_reward: float = 0.0
        self._match_msg: str = ""

    def env_id(self) -> str:
        return f"glyphbench/classics-memorymatch-{self._difficulty}-v0"

    def _reset(self, seed: int) -> GridObservation:
        # Create pairs
        values = list(range(self._num_pairs)) * 2
        self.rng.shuffle(values)
        self._board = list(values)
        self._matched = [False] * self._total_cells
        self._first_flip = None
        self._second_flip = None
        self._on_first_flip = True
        self._pairs_found = 0
        self._total_reward = 0.0
        self._match_msg = ""
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        idx = action  # action index = cell index

        # Validate: skip already matched or out-of-range
        if idx >= self._total_cells or self._matched[idx]:
            return self._render_current_observation(), 0.0, False, False, info

        if self._on_first_flip:
            # First flip
            self._first_flip = idx
            self._second_flip = None
            self._on_first_flip = False
            # Show the first flipped card
            return self._render_current_observation(), 0.0, False, False, info
        else:
            # Second flip
            if idx == self._first_flip:
                # Same card, treat as noop - stay on second flip
                return self._render_current_observation(), 0.0, False, False, info

            self._second_flip = idx
            reward = 0.0
            matched = self._board[self._first_flip] == self._board[idx]

            # Check match
            if matched:
                self._matched[self._first_flip] = True
                self._matched[idx] = True
                self._pairs_found += 1
                reward = 1.0

            # Build message before clearing state
            v1 = PAIR_LETTERS[self._board[self._first_flip]]
            v2 = PAIR_LETTERS[self._board[idx]]
            if matched:
                self._match_msg = f"Match! Both cards show {v1}."
            else:
                self._match_msg = f"No match: {v1} vs {v2}. Cards flipped back."

            # Render with both cards visible before flipping back
            obs = self._render_current_observation()

            # Reset flip state for next turn
            self._match_msg = ""
            self._first_flip = None
            self._second_flip = None
            self._on_first_flip = True

            # Check win
            terminated = self._pairs_found == self._num_pairs
            if terminated:
                reward += 1.0  # bonus for completing

            self._total_reward += reward
            info["pairs_found"] = self._pairs_found
            return obs, reward, terminated, False, info

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(self._cols, self._rows, SYM_FACE_DOWN)

        for i in range(self._total_cells):
            r = i // self._cols
            c = i % self._cols
            if self._matched[i]:
                grid[r][c] = PAIR_LETTERS[self._board[i]]
            elif i == self._first_flip or i == self._second_flip:
                grid[r][c] = PAIR_LETTERS[self._board[i]]

        legend_map: dict[str, str] = {
            SYM_FACE_DOWN: "face-down card",
        }
        for i in range(self._num_pairs):
            legend_map[PAIR_LETTERS[i]] = f"pair {PAIR_LETTERS[i]}"

        flip_phase = "first" if self._on_first_flip else "second"
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Pairs found: {self._pairs_found} / {self._num_pairs}    "
            f"Flip: {flip_phase}"
        )

        msg = ""
        if self._pairs_found == self._num_pairs:
            msg = "All pairs matched! You win!"
        elif self._match_msg:
            msg = self._match_msg
        elif not self._on_first_flip and self._first_flip is not None:
            msg = f"Card at position {self._first_flip} shows {PAIR_LETTERS[self._board[self._first_flip]]}. Now pick the second card."

        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(legend_map),
            hud=hud,
            message=msg,
        )

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Find all matching pairs of cards by flipping two cards per turn.\n\n"
            "RULES\n"
            f"- The board is {self._rows}x{self._cols} with {self._num_pairs} pairs of cards.\n"
            "- Each turn has two phases: flip your first card, then flip your second card.\n"
            "- If both cards match, they stay face up permanently (+1 reward).\n"
            "- If they don't match, both flip back face down.\n"
            "- Cards are numbered 0 to " + str(self._total_cells - 1) + " "
            "(left-to-right, top-to-bottom).\n"
            "- Position 0 is top-left. Position " + str(self._cols - 1) + " is top-right.\n"
            "- Flipping an already-matched card does nothing.\n"
            "- +1 bonus reward for completing all pairs.\n"
            "- Remember which cards you've seen to find matches efficiently!\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------

class MemoryMatchEasyEnv(_MemoryMatchBase):
    _rows = 4
    _cols = 4
    _difficulty = "easy"

    def __init__(self, max_turns: int = 50) -> None:
        super().__init__(max_turns=max_turns)


class MemoryMatchHardEnv(_MemoryMatchBase):
    _rows = 6
    _cols = 6
    _difficulty = "hard"

    def __init__(self, max_turns: int = 120) -> None:
        super().__init__(max_turns=max_turns)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_env(
    "glyphbench/classics-memorymatch-easy-v0",
    "glyphbench.envs.classics.memory_match:MemoryMatchEasyEnv",
)
register_env(
    "glyphbench/classics-memorymatch-hard-v0",
    "glyphbench.envs.classics.memory_match:MemoryMatchHardEnv",
)
