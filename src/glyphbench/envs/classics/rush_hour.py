"""Rush Hour sliding-vehicle puzzle.

Gym IDs:
  glyphbench/classics-rushhour-easy-v0   (4 other vehicles)
  glyphbench/classics-rushhour-hard-v0   (8 other vehicles)
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

BOARD_SIZE = 6
EXIT_ROW = 2  # 0-indexed row where the player car sits and exit is

SYM_EMPTY = "\u00b7"   # ·
SYM_BORDER = "\u2588"  # █
SYM_EXIT = "\u21e8"     # ⇨
SYM_PLAYER = "\u25c6"  # ◆

VEHICLE_LETTERS = tuple("ABCDEFGHIJ")


# ---------------------------------------------------------------------------
# Vehicle representation
# ---------------------------------------------------------------------------

class _Vehicle:
    """A vehicle on the Rush Hour board."""

    __slots__ = ("vid", "row", "col", "length", "horizontal")

    def __init__(self, vid: int, row: int, col: int, length: int, horizontal: bool):
        self.vid = vid
        self.row = row
        self.col = col
        self.length = length
        self.horizontal = horizontal

    def cells(self) -> list[tuple[int, int]]:
        """Return list of (row, col) cells occupied."""
        if self.horizontal:
            return [(self.row, self.col + i) for i in range(self.length)]
        else:
            return [(self.row + i, self.col) for i in range(self.length)]

    def copy(self) -> _Vehicle:
        return _Vehicle(self.vid, self.row, self.col, self.length, self.horizontal)


# ---------------------------------------------------------------------------
# BFS solver (used to verify solvability)
# ---------------------------------------------------------------------------

def _state_key(vehicles: list[_Vehicle]) -> tuple[tuple[int, int], ...]:
    return tuple((v.row, v.col) for v in vehicles)


def _is_solved(vehicles: list[_Vehicle]) -> bool:
    """Player car (vid=0) has right end at col BOARD_SIZE (exited)."""
    p = vehicles[0]
    return p.col + p.length >= BOARD_SIZE


def _occupied_set(vehicles: list[_Vehicle]) -> set[tuple[int, int]]:
    s: set[tuple[int, int]] = set()
    for v in vehicles:
        s.update(v.cells())
    return s


def _bfs_solvable(vehicles: list[_Vehicle], max_nodes: int = 50_000) -> bool:
    """BFS to check if the puzzle is solvable within max_nodes expansions."""
    start = _state_key(vehicles)
    if _is_solved(vehicles):
        return True
    visited: set[tuple[tuple[int, int], ...]] = {start}
    queue: deque[list[_Vehicle]] = deque()
    queue.append([v.copy() for v in vehicles])

    while queue and len(visited) < max_nodes:
        current = queue.popleft()
        occ = _occupied_set(current)

        for idx, veh in enumerate(current):
            for delta in (-1, 1):
                nv = veh.copy()
                if veh.horizontal:
                    nv.col += delta
                else:
                    nv.row += delta
                new_cells = nv.cells()
                # Check bounds
                valid = True
                for r, c in new_cells:
                    if r < 0 or r >= BOARD_SIZE or c < 0 or c >= BOARD_SIZE:
                        # Allow player car to exit right
                        if idx == 0 and veh.horizontal and c >= BOARD_SIZE:
                            continue
                        valid = False
                        break
                if not valid:
                    continue
                # Check collisions with other vehicles
                old_cells = set(veh.cells())
                collision = False
                for r, c in new_cells:
                    if (r, c) in occ and (r, c) not in old_cells:
                        collision = True
                        break
                if collision:
                    continue

                # Apply move
                new_state = [v.copy() for v in current]
                new_state[idx] = nv
                if _is_solved(new_state):
                    return True
                key = _state_key(new_state)
                if key not in visited:
                    visited.add(key)
                    queue.append(new_state)

    return False


# ---------------------------------------------------------------------------
# Puzzle generation
# ---------------------------------------------------------------------------

def _generate_puzzle(rng, num_other_vehicles: int) -> list[_Vehicle]:
    """Generate a solvable Rush Hour puzzle with the given number of other vehicles."""
    for _ in range(500):
        vehicles: list[_Vehicle] = []
        # Player car: horizontal, length 2, on EXIT_ROW
        start_col = rng.integers(0, 3)  # 0..2 so car fits
        player = _Vehicle(0, EXIT_ROW, int(start_col), 2, True)
        vehicles.append(player)

        occupied = set(player.cells())
        placed = 0
        attempts = 0
        while placed < num_other_vehicles and attempts < 200:
            attempts += 1
            length = int(rng.choice([2, 2, 2, 3]))  # mostly cars, some trucks
            horizontal = bool(rng.choice([True, False]))
            if horizontal:
                max_col = BOARD_SIZE - length
                max_row = BOARD_SIZE - 1
            else:
                max_col = BOARD_SIZE - 1
                max_row = BOARD_SIZE - length
            row = int(rng.integers(0, max_row + 1))
            col = int(rng.integers(0, max_col + 1))
            v = _Vehicle(placed + 1, row, col, length, horizontal)
            cells = v.cells()
            if any(c in occupied for c in cells):
                continue
            # Don't completely block exit row exit cell unless intentional
            vehicles.append(v)
            occupied.update(cells)
            placed += 1

        if placed < num_other_vehicles:
            continue

        if _bfs_solvable(vehicles):
            return vehicles

    # Fallback: trivial puzzle (just the player car near the exit)
    return [_Vehicle(0, EXIT_ROW, BOARD_SIZE - 2, 2, True)]


# ---------------------------------------------------------------------------
# Build action spec dynamically
# ---------------------------------------------------------------------------

def _build_action_spec(num_vehicles: int) -> ActionSpec:
    """Build action spec for N vehicles (0 = player + 1..N-1 others)."""
    names: list[str] = []
    descs: list[str] = []
    for i in range(num_vehicles):
        label = "PLAYER" if i == 0 else VEHICLE_LETTERS[i - 1]
        names.append(f"MOVE_{label}_FWD")
        descs.append(f"slide {label} forward (right if horizontal, down if vertical)")
        names.append(f"MOVE_{label}_BACK")
        descs.append(f"slide {label} backward (left if horizontal, up if vertical)")
    names.append("NOOP")
    descs.append("do nothing")
    return ActionSpec(names=tuple(names), descriptions=tuple(descs))


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

class _RushHourBase(BaseAsciiEnv):
    """Slide vehicles to let your car exit the right edge."""

    noop_action_name: str = "NOOP"

    _num_other: int = 4
    _difficulty: str = "easy"

    def __init__(self, max_turns: int = 200) -> None:
        # Total vehicles = 1 (player) + _num_other
        self._total_vehicles = 1 + self._num_other
        self.action_spec = _build_action_spec(self._total_vehicles)
        super().__init__(max_turns=max_turns)
        self._vehicles: list[_Vehicle] = []
        self._solved: bool = False

    def env_id(self) -> str:
        return f"glyphbench/classics-rushhour-{self._difficulty}-v0"

    def _reset(self, seed: int) -> GridObservation:
        self._vehicles = _generate_puzzle(self.rng, self._num_other)
        # Rebuild action spec if puzzle has fewer vehicles than expected
        actual = len(self._vehicles)
        if actual != self._total_vehicles:
            self._total_vehicles = actual
            self.action_spec = _build_action_spec(self._total_vehicles)
            self.action_space.__init__(self.action_spec.n)  # type: ignore[misc]
        self._solved = False
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        name = self.action_spec.names[action]

        if name == "NOOP":
            return self._render_current_observation(), -0.01, False, False, info

        # Parse action: MOVE_{LABEL}_{DIR}
        parts = name.split("_")
        # parts = ["MOVE", label, dir]  but label could be "PLAYER" or a letter
        direction = parts[-1]  # FWD or BACK
        label = "_".join(parts[1:-1])

        # Find vehicle index
        if label == "PLAYER":
            vid = 0
        else:
            vid = VEHICLE_LETTERS.index(label) + 1

        if vid >= len(self._vehicles):
            # Invalid vehicle reference
            return self._render_current_observation(), -0.01, False, False, info

        veh = self._vehicles[vid]
        delta = 1 if direction == "FWD" else -1

        # Try to move
        nv = veh.copy()
        if veh.horizontal:
            nv.col += delta
        else:
            nv.row += delta

        new_cells = nv.cells()
        old_cells = set(veh.cells())
        occ = _occupied_set(self._vehicles)

        valid = True
        for r, c in new_cells:
            if r < 0 or r >= BOARD_SIZE or c < 0 or c >= BOARD_SIZE:
                # Allow player to exit right
                if vid == 0 and veh.horizontal and c >= BOARD_SIZE:
                    continue
                valid = False
                break
            if (r, c) in occ and (r, c) not in old_cells:
                valid = False
                break

        if valid:
            self._vehicles[vid] = nv
            # Check win
            if vid == 0 and nv.col + nv.length >= BOARD_SIZE:
                self._solved = True
                return self._render_current_observation(), 1.0, True, False, info

        return self._render_current_observation(), -0.01, False, False, info

    def _render_current_observation(self) -> GridObservation:
        # Grid: BOARD_SIZE + 2 for borders, plus exit indicator
        w = BOARD_SIZE + 3  # border + 6 cells + border + exit column
        h = BOARD_SIZE + 2  # border + 6 cells + border
        grid = make_empty_grid(w, h, SYM_EMPTY)

        # Draw borders
        for x in range(w):
            grid[0][x] = SYM_BORDER
            grid[h - 1][x] = SYM_BORDER
        for y in range(h):
            grid[y][0] = SYM_BORDER
            grid[y][w - 1] = SYM_BORDER

        # Exit marker on right side at EXIT_ROW
        exit_y = EXIT_ROW + 1  # +1 for border
        grid[exit_y][w - 2] = SYM_EXIT
        grid[exit_y][w - 1] = SYM_EXIT

        # Place vehicles
        for v in self._vehicles:
            sym = SYM_PLAYER if v.vid == 0 else VEHICLE_LETTERS[v.vid - 1]
            for r, c in v.cells():
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    grid[r + 1][c + 1] = sym  # +1 for border offset

        legend_map: dict[str, str] = {
            SYM_PLAYER: "your car (must exit right)",
            SYM_EMPTY: "empty cell",
            SYM_BORDER: "border wall",
            SYM_EXIT: "exit",
        }
        for v in self._vehicles:
            if v.vid > 0 and v.vid - 1 < len(VEHICLE_LETTERS):
                letter = VEHICLE_LETTERS[v.vid - 1]
                orient = "horizontal" if v.horizontal else "vertical"
                kind = "car" if v.length == 2 else "truck"
                legend_map[letter] = f"{orient} {kind} (length {v.length})"

        hud = f"Step: {self._turn} / {self.max_turns}    Vehicles: {len(self._vehicles)}"
        msg = ""
        if self._solved:
            msg = "Your car has exited! Puzzle solved!"

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
            "Slide vehicles on a 6x6 grid to clear a path for your car to exit "
            "through the right edge. This is the classic Rush Hour puzzle.\n\n"
            "RULES\n"
            "- Your car (marked with \u25c6) is on row 3 and must exit through the right edge (\u21e8).\n"
            "- Each vehicle can only slide along its orientation (horizontal = left/right, "
            "vertical = up/down).\n"
            "- Vehicles cannot pass through each other.\n"
            "- FWD = right for horizontal vehicles, down for vertical vehicles.\n"
            "- BACK = left for horizontal vehicles, up for vertical vehicles.\n"
            "- Each move costs -0.01 reward. Exiting gives +1 reward.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------

class RushHourEasyEnv(_RushHourBase):
    _num_other = 4
    _difficulty = "easy"

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)


class RushHourHardEnv(_RushHourBase):
    _num_other = 8
    _difficulty = "hard"

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_env(
    "glyphbench/classics-rushhour-easy-v0",
    "glyphbench.envs.classics.rush_hour:RushHourEasyEnv",
)
register_env(
    "glyphbench/classics-rushhour-hard-v0",
    "glyphbench.envs.classics.rush_hour:RushHourHardEnv",
)
