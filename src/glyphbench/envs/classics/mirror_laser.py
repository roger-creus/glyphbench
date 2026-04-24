"""Mirror laser puzzle: rotate mirrors to direct a laser beam to the target.

Gym IDs:
  glyphbench/classics-mirrorlaser-v0  (7x7)
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE = 7
TOTAL_CELLS = GRID_SIZE * GRID_SIZE

_action_names = tuple(f"ROTATE_{i}" for i in range(TOTAL_CELLS))
_action_descs = tuple(
    f"rotate mirror at row {i // GRID_SIZE}, col {i % GRID_SIZE} by 90 degrees CW (no-op if no mirror)"
    for i in range(TOTAL_CELLS)
)

MIRROR_ACTION_SPEC = ActionSpec(names=_action_names, descriptions=_action_descs)

# Directions: (dx, dy)
DIR_RIGHT = (1, 0)
DIR_LEFT = (-1, 0)
DIR_UP = (0, -1)
DIR_DOWN = (0, 1)

# Mirror types and their reflections
# ╱ (forward slash mirror): reflects RIGHT->UP, LEFT->DOWN, UP->RIGHT, DOWN->LEFT
# ╲ (backslash mirror):     reflects RIGHT->DOWN, LEFT->UP, UP->LEFT, DOWN->RIGHT

MIRROR_FWD = "\u2571"  # ╱
MIRROR_BWD = "\u2572"  # ╲

_REFLECT_FWD: dict[tuple[int, int], tuple[int, int]] = {
    DIR_RIGHT: DIR_UP,
    DIR_LEFT: DIR_DOWN,
    DIR_UP: DIR_RIGHT,
    DIR_DOWN: DIR_LEFT,
}

_REFLECT_BWD: dict[tuple[int, int], tuple[int, int]] = {
    DIR_RIGHT: DIR_DOWN,
    DIR_LEFT: DIR_UP,
    DIR_UP: DIR_LEFT,
    DIR_DOWN: DIR_RIGHT,
}

SYM_SOURCE = "\u25b8"  # ▸
SYM_TARGET = "\u2605"  # ★
SYM_EMPTY = "\u00b7"   # ·
SYM_BEAM_H = "\u2500"  # ─
SYM_BEAM_V = "\u2502"  # │

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class MirrorLaserEnv(BaseGlyphEnv):
    """Place/rotate mirrors to direct a laser beam to the target."""

    action_spec = MIRROR_ACTION_SPEC
    noop_action_name: str = "ROTATE_0"

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        # Grid of mirror types: None = empty, MIRROR_FWD or MIRROR_BWD
        self._mirrors: list[list[str | None]] = []
        self._source_pos: tuple[int, int] = (0, 0)  # (x, y) on left edge
        self._target_pos: tuple[int, int] = (0, 0)
        self._beam_path: list[tuple[int, int, tuple[int, int]]] = []  # (x, y, dir)

    def env_id(self) -> str:
        return "glyphbench/classics-mirrorlaser-v0"

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_puzzle(self) -> None:
        """Generate a mirror puzzle with a solvable configuration."""
        s = GRID_SIZE
        self._mirrors = [[None for _ in range(s)] for _ in range(s)]

        # Laser source on left edge pointing right
        src_row = int(self.rng.integers(0, s))
        self._source_pos = (0, src_row)

        # Target at a random interior position (not on left edge)
        for _ in range(100):
            tx = int(self.rng.integers(2, s))
            ty = int(self.rng.integers(0, s))
            if (tx, ty) != self._source_pos:
                self._target_pos = (tx, ty)
                break

        # Generate a valid solution path from source to target, then place mirrors
        # Strategy: trace from source going right, place mirrors to redirect beam to target
        self._place_solution_mirrors()

        # Add some extra mirrors as distractors
        n_extra = int(self.rng.integers(2, 5))
        empty_cells = [
            (x, y) for y in range(s) for x in range(s)
            if self._mirrors[y][x] is None
            and (x, y) != self._source_pos
            and (x, y) != self._target_pos
        ]
        if empty_cells:
            self.rng.shuffle(empty_cells)  # type: ignore[arg-type]
            for i in range(min(n_extra, len(empty_cells))):
                ex, ey = empty_cells[i]
                self._mirrors[ey][ex] = MIRROR_FWD if self.rng.random() < 0.5 else MIRROR_BWD

        # Scramble all mirrors (randomly flip each one)
        for y in range(s):
            for x in range(s):
                if self._mirrors[y][x] is not None:
                    if self.rng.random() < 0.5:
                        self._mirrors[y][x] = (
                            MIRROR_BWD if self._mirrors[y][x] == MIRROR_FWD else MIRROR_FWD
                        )

    def _place_solution_mirrors(self) -> None:
        """Place mirrors that create a valid path from source to target."""
        sx, sy = self._source_pos
        tx, ty = self._target_pos

        # Simple strategy: go right from source, place a mirror to redirect
        # vertically, then place another mirror to redirect horizontally to target.

        # First mirror: somewhere on the source row, redirect up/down toward target row
        mid_x = int(self.rng.integers(max(1, min(sx + 1, tx)), max(sx + 1, tx) + 1))
        mid_x = min(mid_x, GRID_SIZE - 1)

        if ty < sy:
            # Need to go up: beam comes from left (RIGHT), needs to go UP -> use ╱
            self._mirrors[sy][mid_x] = MIRROR_FWD
            # Second mirror at (mid_x, ty): beam comes from DOWN, needs to go RIGHT -> use ╱
            self._mirrors[ty][mid_x] = MIRROR_FWD
        elif ty > sy:
            # Need to go down: beam comes from left (RIGHT), needs to go DOWN -> use ╲
            self._mirrors[sy][mid_x] = MIRROR_BWD
            # Second mirror at (mid_x, ty): beam comes from UP, needs to go RIGHT -> use ╲
            self._mirrors[ty][mid_x] = MIRROR_BWD
        # If same row, no mirrors needed (beam goes straight)

        # If target is not directly to the right of last mirror, add one more redirect
        if ty != sy and mid_x != tx:
            # We might need another mirror at (tx, ty) area
            # The beam after second mirror goes RIGHT. If target x != mid_x, we need more.
            # Keep it simple: target is to the right, beam reaches it eventually.
            pass

    # ------------------------------------------------------------------
    # Beam tracing
    # ------------------------------------------------------------------

    def _trace_beam(self) -> tuple[list[tuple[int, int, tuple[int, int]]], bool]:
        """Trace the laser beam. Returns (path, hit_target)."""
        s = GRID_SIZE
        path: list[tuple[int, int, tuple[int, int]]] = []
        x, y = self._source_pos
        d = DIR_RIGHT
        visited: set[tuple[int, int, int, int]] = set()

        # Start one step into the grid from source
        x += d[0]
        y += d[1]

        max_steps = s * s * 4  # prevent infinite loops
        for _ in range(max_steps):
            if x < 0 or x >= s or y < 0 or y >= s:
                break

            state = (x, y, d[0], d[1])
            if state in visited:
                break  # loop detected
            visited.add(state)

            path.append((x, y, d))

            if (x, y) == self._target_pos:
                return path, True

            mirror = self._mirrors[y][x]
            if mirror == MIRROR_FWD:
                d = _REFLECT_FWD[d]
            elif mirror == MIRROR_BWD:
                d = _REFLECT_BWD[d]

            x += d[0]
            y += d[1]

        return path, False

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._generate_puzzle()
        self._beam_path, _ = self._trace_beam()
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        r, c = divmod(action, GRID_SIZE)

        # Rotate mirror at (c, r) if it exists
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            mirror = self._mirrors[r][c]
            if mirror is not None:
                # Toggle between FWD and BWD (90 degree rotation)
                self._mirrors[r][c] = MIRROR_BWD if mirror == MIRROR_FWD else MIRROR_FWD

        # Trace beam
        self._beam_path, hit_target = self._trace_beam()

        reward = 0.0
        terminated = False

        if hit_target:
            reward = 1.0
            terminated = True
            info["target_hit"] = True

        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        s = GRID_SIZE
        grid = make_empty_grid(s, s, SYM_EMPTY)

        # Draw beam path (before mirrors so mirrors overwrite)
        for bx, by, bd in self._beam_path:
            if (bx, by) != self._target_pos and self._mirrors[by][bx] is None:
                if bd[0] != 0:  # horizontal
                    grid[by][bx] = SYM_BEAM_H
                else:  # vertical
                    grid[by][bx] = SYM_BEAM_V

        # Mirrors
        for y in range(s):
            for x in range(s):
                if self._mirrors[y][x] is not None:
                    grid[y][x] = self._mirrors[y][x]

        # Source
        sx, sy = self._source_pos
        grid[sy][sx] = SYM_SOURCE

        # Target
        tx, ty = self._target_pos
        grid[ty][tx] = SYM_TARGET

        legend = build_legend({
            SYM_SOURCE: "laser source (points right)",
            SYM_TARGET: "target (direct beam here)",
            MIRROR_FWD: "mirror / (reflects beam)",
            MIRROR_BWD: "mirror \\ (reflects beam)",
            SYM_BEAM_H: "laser beam (horizontal)",
            SYM_BEAM_V: "laser beam (vertical)",
            SYM_EMPTY: "empty cell",
        })

        _, hit = self._trace_beam()
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Source: row {sy}    "
            f"Target: ({tx}, {ty})    "
            f"Beam hits target: {'YES' if hit else 'no'}"
        )

        return GridObservation(grid=grid_to_string(grid), legend=legend, hud=hud, message="")

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Rotate mirrors to direct a laser beam from the source to the target.\n\n"
            "RULES\n"
            f"- The grid is {GRID_SIZE}x{GRID_SIZE}.\n"
            "- The laser source is on the left edge, emitting a beam to the right.\n"
            "- Mirrors reflect the beam:\n"
            "  - / mirror: reflects right->up, left->down, up->right, down->left\n"
            "  - \\ mirror: reflects right->down, left->up, up->left, down->right\n"
            "- Each ROTATE action toggles a mirror between / and \\ orientation.\n"
            "- Rotating a cell with no mirror does nothing.\n"
            f"- Actions: ROTATE_0 through ROTATE_{TOTAL_CELLS - 1} where index = row * {GRID_SIZE} + col.\n"
            "- Win by directing the beam to hit the target (+1 reward).\n"
            "- The beam path is shown on the grid as horizontal/vertical lines.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

