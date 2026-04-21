"""MiniHack Boxoban environments. Sokoban puzzles in a dungeon."""

from __future__ import annotations

from typing import Any

from glyphbench.core.ascii_primitives import build_legend, grid_to_string, make_empty_grid
from glyphbench.core.observation import GridObservation
from glyphbench.envs.minihack.base import MOVE_VECTORS, MiniHackBase


class _BoxobanBase(MiniHackBase):
    """Base for Boxoban (Sokoban) puzzle environments.

    The agent must push all boxes (``0``) onto target positions (``X``).
    Walking into a box pushes it in the same direction, provided the cell
    behind the box is empty floor and not another box or wall.
    """

    _grid_size: int = 7
    _num_boxes: int = 2

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._box_positions: list[tuple[int, int]] = []
        self._target_positions: list[tuple[int, int]] = []

    def _generate_level(self, seed: int) -> None:
        s = self._grid_size
        self._init_grid(s, s)
        self._box_positions = []
        self._target_positions = []

        occupied: set[tuple[int, int]] = set()

        # Player
        px = int(self.rng.integers(1, s - 1))
        py = int(self.rng.integers(1, s - 1))
        self._place_player(px, py)
        occupied.add((px, py))

        # Place boxes and targets
        for _ in range(self._num_boxes):
            while True:
                bx = int(self.rng.integers(2, s - 2))
                by = int(self.rng.integers(2, s - 2))
                if (bx, by) not in occupied:
                    break
            self._box_positions.append((bx, by))
            occupied.add((bx, by))

            while True:
                tx = int(self.rng.integers(1, s - 1))
                ty = int(self.rng.integers(1, s - 1))
                if (tx, ty) not in occupied:
                    break
            self._target_positions.append((tx, ty))
            occupied.add((tx, ty))

    # ------------------------------------------------------------------
    # Override _step for box-pushing mechanics
    # ------------------------------------------------------------------

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        self._message = ""

        if name in MOVE_VECTORS:
            dx, dy = MOVE_VECTORS[name]
            px, py = self._player_pos
            nx, ny = px + dx, py + dy

            # Check if pushing a box
            box_idx: int | None = None
            for i, (bx, by) in enumerate(self._box_positions):
                if (bx, by) == (nx, ny):
                    box_idx = i
                    break

            if box_idx is not None:
                # Try to push box
                behind_x, behind_y = nx + dx, ny + dy
                if self._is_walkable(behind_x, behind_y) and (
                    behind_x,
                    behind_y,
                ) not in self._box_positions:
                    self._box_positions[box_idx] = (behind_x, behind_y)
                    self._player_pos = (nx, ny)
                    self._message = "You push the box."
                # else: blocked, can't push
            elif self._is_walkable(nx, ny):
                self._player_pos = (nx, ny)
        # Other actions: no-op for Boxoban

        # Check win condition: all boxes on targets
        terminated = False
        reward = 0.0
        if set(self._box_positions) == set(self._target_positions):
            terminated = True
            reward = 1.0
            self._message = "All boxes on targets! Puzzle complete!"

        info: dict[str, Any] = {
            "player_pos": self._player_pos,
            "boxes_on_target": sum(
                1 for b in self._box_positions if b in self._target_positions
            ),
        }
        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        render = make_empty_grid(self._grid_w, self._grid_h, fill=" ")
        symbols: dict[str, str] = {}

        for y in range(self._grid_h):
            for x in range(self._grid_w):
                ch = self._grid[y][x]
                render[y][x] = ch
                if ch == "·":
                    symbols["·"] = "floor"
                elif ch in ("-", "|", "█"):
                    symbols[ch] = "wall"

        # Targets (only visible when no box sits on them)
        for tx, ty in self._target_positions:
            if (tx, ty) not in self._box_positions:
                render[ty][tx] = "X"
        symbols["X"] = "target"

        # Boxes
        for bx, by in self._box_positions:
            if (bx, by) in self._target_positions:
                render[by][bx] = "*"  # box on target
                symbols["*"] = "box on target"
            else:
                render[by][bx] = "0"
                symbols["0"] = "box"

        # Player
        px, py = self._player_pos
        render[py][px] = "@"
        symbols["@"] = "you"

        hud = (
            f"Turn: {self._turn}    Pos: ({px},{py})    "
            f"Boxes on target: "
            f"{sum(1 for b in self._box_positions if b in self._target_positions)}"
            f"/{self._num_boxes}"
        )
        return GridObservation(
            grid=grid_to_string(render),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    # ------------------------------------------------------------------
    # Task description
    # ------------------------------------------------------------------

    def _task_description(self) -> str:
        return (
            f"Sokoban puzzle: push {self._num_boxes} boxes (0) onto target "
            f"positions (X). Walk into a box to push it. Boxes can't be pulled. "
            f"A box on its target shows as *. "
            f"Reward: +1 when all boxes are on targets."
        )


class MiniHackBoxobanMediumEnv(_BoxobanBase):
    """7x7 grid, 2 boxes."""

    _grid_size = 7
    _num_boxes = 2

    def env_id(self) -> str:
        return "glyphbench/minihack-boxoban-medium-v0"


class MiniHackBoxobanHardEnv(_BoxobanBase):
    """9x9 grid, 3 boxes."""

    _grid_size = 9
    _num_boxes = 3

    def env_id(self) -> str:
        return "glyphbench/minihack-boxoban-hard-v0"


class MiniHackBoxobanUnfilteredEnv(_BoxobanBase):
    """9x9 grid, 3 boxes, randomly generated (may not be solvable)."""

    _grid_size = 9
    _num_boxes = 3

    def env_id(self) -> str:
        return "glyphbench/minihack-boxoban-unfiltered-v0"
