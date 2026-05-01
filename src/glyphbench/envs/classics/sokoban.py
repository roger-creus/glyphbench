"""Sokoban: push boxes onto targets.

Gym IDs:
  glyphbench/classics-sokoban-easy-v0    (7x7, 1-2 boxes)
  glyphbench/classics-sokoban-medium-v0  (9x9, 3 boxes)
  glyphbench/classics-sokoban-hard-v0    (11x11, 4-5 boxes)
"""

from __future__ import annotations

from collections import deque
from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOKOBAN_ACTION_SPEC = ActionSpec(
    names=("UP", "DOWN", "LEFT", "RIGHT"),
    descriptions=(
        "move/push up",
        "move/push down",
        "move/push left",
        "move/push right",
    ),
)

SYM_PLAYER = "@"
SYM_BOX = "\u25fc"       # ◼
SYM_TARGET = "\u25ce"     # ◎
SYM_BOX_ON_TARGET = "\u25c8"  # ◈
SYM_WALL = "\u2588"       # █
SYM_FLOOR = "\u00b7"      # ·

_DIR_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

# ---------------------------------------------------------------------------
# Base Sokoban Env
# ---------------------------------------------------------------------------


class _SokobanBase(BaseGlyphEnv):
    """Sokoban: push all boxes onto target positions."""

    action_spec = SOKOBAN_ACTION_SPEC
    noop_action_name: str = "UP"

    _grid_size: int = 7
    _num_boxes: int = 2
    _difficulty: str = "easy"

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._walls: set[tuple[int, int]] = set()
        self._boxes: set[tuple[int, int]] = set()
        self._targets: set[tuple[int, int]] = set()
        self._player: tuple[int, int] = (0, 0)
        self._boxes_on_targets: int = 0

    def env_id(self) -> str:
        return f"glyphbench/classics-sokoban-{self._difficulty}-v0"

    # ------------------------------------------------------------------
    # Puzzle generation
    # ------------------------------------------------------------------

    def _generate_puzzle(self) -> None:
        """Generate a solvable Sokoban puzzle via reverse placement.

        Strategy: place player and targets randomly in the interior, then
        simulate pulling boxes from targets to random positions. This
        guarantees the puzzle is solvable because we constructed it by
        reverse-solving.
        """
        s = self._grid_size
        n = self._num_boxes

        # Interior cells (inside walls)
        interior = [(x, y) for x in range(1, s - 1) for y in range(1, s - 1)]

        # Walls around border
        self._walls = set()
        for x in range(s):
            self._walls.add((x, 0))
            self._walls.add((x, s - 1))
        for y in range(s):
            self._walls.add((0, y))
            self._walls.add((s - 1, y))

        # Add some random interior walls for variety (but keep it sparse)
        num_interior_walls = int(self.rng.integers(0, max(1, (s * s) // 20)))
        shuffled = list(interior)
        self.rng.shuffle(shuffled)

        wall_candidates = shuffled[:num_interior_walls + n + 1 + n]
        # Reserve first n+1+n spots for targets, player, box positions

        # Place targets
        targets_list = wall_candidates[:n]
        self._targets = set(targets_list)
        remaining = wall_candidates[n:]

        # Place player
        player_pos = remaining[0]
        self._player = player_pos
        remaining = remaining[1:]

        # Initial box positions (start on targets, then we "pull" them away)
        # For simplicity we start boxes on targets and simulate reverse moves
        self._boxes = set(targets_list)

        # Now "pull" each box off its target by simulating reverse moves
        # (player pushes box, in reverse = player pulls box toward itself)
        pulled_positions: set[tuple[int, int]] = set()
        occupied = self._walls | {self._player}

        for target in targets_list:
            bx, by = target
            moved = False
            # Try random directions to pull the box
            dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            dir_order = list(range(4))
            self.rng.shuffle(dir_order)

            for di in dir_order:
                ddx, ddy = dirs[di]
                # To "pull" box in direction (ddx, ddy), player must be on
                # the opposite side: (bx - ddx, by - ddy) and box moves to
                # (bx + ddx, by + ddy)
                new_bx, new_by = bx + ddx, by + ddy
                pull_from = (bx - ddx, by - ddy)

                if (1 <= new_bx < s - 1 and 1 <= new_by < s - 1
                        and (new_bx, new_by) not in occupied
                        and (new_bx, new_by) not in pulled_positions
                        and (new_bx, new_by) not in self._targets
                        and 1 <= pull_from[0] < s - 1
                        and 1 <= pull_from[1] < s - 1
                        and pull_from not in occupied
                        and pull_from not in pulled_positions):
                    self._boxes.discard((bx, by))
                    self._boxes.add((new_bx, new_by))
                    pulled_positions.add((new_bx, new_by))
                    occupied.add((new_bx, new_by))
                    moved = True
                    break

            if not moved:
                # Box stays on target -- still solvable (it's already solved
                # for this box)
                pulled_positions.add((bx, by))
                occupied.add((bx, by))

        # Place player in a reachable position
        # Find a free cell reachable from any box neighbor
        free_cells = [
            (x, y) for x in range(1, s - 1) for y in range(1, s - 1)
            if (x, y) not in self._walls and (x, y) not in self._boxes
        ]
        if free_cells:
            # BFS to check reachability from a box neighbor
            reachable = self._reachable_cells(free_cells[0], self._walls | self._boxes, s)
            # Pick a random reachable cell for the player
            reachable_list = [c for c in reachable if c not in self._boxes and c not in self._targets]
            if reachable_list:
                idx = int(self.rng.integers(0, len(reachable_list)))
                self._player = reachable_list[idx]
            else:
                self._player = free_cells[0]

        self._boxes_on_targets = len(self._boxes & self._targets)

    @staticmethod
    def _reachable_cells(
        start: tuple[int, int],
        blocked: set[tuple[int, int]],
        grid_size: int,
    ) -> set[tuple[int, int]]:
        """BFS to find all reachable cells from start."""
        visited: set[tuple[int, int]] = {start}
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = cx + ddx, cy + ddy
                if (1 <= nx < grid_size - 1 and 1 <= ny < grid_size - 1
                        and (nx, ny) not in blocked and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return visited

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._generate_puzzle()
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        name = self.action_spec.names[action]
        dx, dy = _DIR_DELTAS[name]
        px, py = self._player
        nx, ny = px + dx, py + dy

        old_on_targets = len(self._boxes & self._targets)

        # Check if target cell is free
        if (nx, ny) in self._walls:
            # Bump into wall -- no move
            pass
        elif (nx, ny) in self._boxes:
            # Pushing a box
            bx, by = nx + dx, ny + dy
            if (bx, by) not in self._walls and (bx, by) not in self._boxes:
                self._boxes.discard((nx, ny))
                self._boxes.add((bx, by))
                self._player = (nx, ny)
        else:
            self._player = (nx, ny)

        new_on_targets = len(self._boxes & self._targets)
        # Pattern A: each box placed/unplaced changes reward by +/-1/N so
        # cumulative reward telescopes to placed_final / N. Solved = 1.0.
        # No completion bonus -- we already cap at 1.0 by construction.
        n_targets = max(1, len(self._targets))
        reward = float(new_on_targets - old_on_targets) / n_targets
        self._boxes_on_targets = new_on_targets

        terminated = new_on_targets == len(self._targets)

        info["boxes_on_targets"] = new_on_targets
        info["total_targets"] = len(self._targets)
        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        s = self._grid_size
        grid = make_empty_grid(s, s, SYM_FLOOR)

        for wx, wy in self._walls:
            grid[wy][wx] = SYM_WALL

        for tx, ty in self._targets:
            if (tx, ty) not in self._boxes:
                grid[ty][tx] = SYM_TARGET

        for bx, by in self._boxes:
            if (bx, by) in self._targets:
                grid[by][bx] = SYM_BOX_ON_TARGET
            else:
                grid[by][bx] = SYM_BOX

        px, py = self._player
        grid[py][px] = SYM_PLAYER

        legend = build_legend({
            SYM_PLAYER: "player (you)",
            SYM_BOX: "box",
            SYM_TARGET: "target",
            SYM_BOX_ON_TARGET: "box on target",
            SYM_WALL: "wall",
            SYM_FLOOR: "floor",
        })

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Boxes on targets: {self._boxes_on_targets} / {len(self._targets)}"
        )

        return GridObservation(grid=grid_to_string(grid), legend=legend, hud=hud, message="")

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Push all boxes onto target positions. You can push a box by walking "
            "into it, but only if the space behind the box is free. You cannot pull boxes.\n\n"
            "RULES\n"
            f"- The grid is {self._grid_size}x{self._grid_size}.\n"
            f"- There are {self._num_boxes} boxes and {self._num_boxes} targets.\n"
            "- Moving into a box pushes it one cell in the same direction, if the cell behind it is empty.\n"
            f"- You get +{1.0 / self._num_boxes:.4f} reward when a box lands on a target, the\n"
            f"  same magnitude negative when a box is pushed off a target.\n"
            "- All boxes on targets ends the episode with cumulative reward = 1.0.\n"
            "- Be careful: pushing a box into a corner may make the puzzle unsolvable.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class SokobanEasyEnv(_SokobanBase):
    _grid_size = 7
    _num_boxes = 2
    _difficulty = "easy"

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)


class SokobanMediumEnv(_SokobanBase):
    _grid_size = 9
    _num_boxes = 3
    _difficulty = "medium"

    def __init__(self, max_turns: int = 300) -> None:
        super().__init__(max_turns=max_turns)


class SokobanHardEnv(_SokobanBase):
    _grid_size = 11
    _num_boxes = 5
    _difficulty = "hard"

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

