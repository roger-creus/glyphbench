"""miniatari Amidar.

Identity: Paint cells of a small rectangle by walking on them while a
patrol enemy roams the path.
Win condition: paint all 10 cells of the rectangle perimeter.
Reward: Pattern D, +1/10 per painted cell; -1 if caught by patrol.

Gym ID: glyphbench/miniatari-amidar-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=118, mean_return=-0.417
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniAmidarEnv(MiniatariBase):
    """Mini Amidar: 12x10 grid, walk a 4x4 rectangle perimeter.

    The painting rectangle is at columns 3..6, rows 3..6 — 12 perimeter
    cells (4 corners + 8 edge cells). Paint each by stepping onto it.
    A patrol enemy (e) traverses the perimeter clockwise 1 cell every
    3 ticks. Pattern D: +1/10 per painted cell, -1 on catch.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN"),
        descriptions=(
            "do nothing",
            "move left and face left",
            "move right and face right",
            "move up and face up",
            "move down and face down",
        ),
    )

    default_max_turns = 300

    _WIDTH = 12
    _HEIGHT = 10
    _RECT = (3, 3, 5, 6)  # x0, y0, x1, y1 (inclusive); 3x4 -> 10 perimeter cells
    _N_CELLS = 10  # perimeter cells to paint
    _WIN_TARGET = _N_CELLS
    _PATROL_MOVE_EVERY = 3

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._painted: set[tuple[int, int]] = set()
        self._patrol_pos: list[int] = [0, 0]
        self._patrol_step: int = 0
        self._tick_count: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-amidar-v0"

    def _perimeter_cells(self) -> list[tuple[int, int]]:
        x0, y0, x1, y1 = self._RECT
        cells: list[tuple[int, int]] = []
        # Top row: (x0..x1, y0)
        for x in range(x0, x1 + 1):
            cells.append((x, y0))
        # Right col: (x1, y0+1..y1)
        for y in range(y0 + 1, y1 + 1):
            cells.append((x1, y))
        # Bottom row reversed: (x1-1..x0, y1)
        for x in range(x1 - 1, x0 - 1, -1):
            cells.append((x, y1))
        # Left col reversed: (x0, y1-1..y0+1)
        for y in range(y1 - 1, y0, -1):
            cells.append((x0, y))
        return cells

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        self._painted = set()
        # Player starts at top-left corner of the rectangle (which is the
        # first un-painted target — but they still need to step on it)
        x0, y0, _x1, _y1 = self._RECT
        self._player_x = x0
        self._player_y = y0 - 1  # one above the rectangle
        self._player_dir = (0, 1)
        # Patrol starts at the opposite (bottom-right) corner
        cells = self._perimeter_cells()
        self._patrol_step = len(cells) // 2
        px, py = cells[self._patrol_step]
        self._patrol_pos = [px, py]

    def _on_perim(self, x: int, y: int) -> bool:
        x0, y0, x1, y1 = self._RECT
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return False
        return x == x0 or x == x1 or y == y0 or y == y1

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        # 1. Player move
        nx, ny = self._player_x, self._player_y
        if action_name == "LEFT":
            nx = max(0, nx - 1)
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            nx = min(self._WIDTH - 1, nx + 1)
            self._player_dir = (1, 0)
        elif action_name == "UP":
            ny = max(0, ny - 1)
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            ny = min(self._HEIGHT - 1, ny + 1)
            self._player_dir = (0, 1)
        self._player_x, self._player_y = nx, ny

        # 2. Paint cell if standing on perimeter and not yet painted
        if self._on_perim(self._player_x, self._player_y):
            cell = (self._player_x, self._player_y)
            if cell not in self._painted:
                self._painted.add(cell)
                reward += self._progress_reward(self._WIN_TARGET)
                self._progress += 1
                self._message = f"Painted! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info

        # 3. Patrol step
        if self._tick_count % self._PATROL_MOVE_EVERY == 0:
            cells = self._perimeter_cells()
            self._patrol_step = (self._patrol_step + 1) % len(cells)
            self._patrol_pos = list(cells[self._patrol_step])

        # 4. Catch?
        if (self._patrol_pos[0] == self._player_x and
                self._patrol_pos[1] == self._player_y):
            self._message = "Caught by the patrol!"
            reward += self._death_reward()
            self._on_life_lost()
            return reward, True, info

        info["progress"] = self._progress
        info["painted"] = len(self._painted)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Render perimeter
        for cx, cy in self._perimeter_cells():
            if (cx, cy) in self._painted:
                grid[cy][cx] = "▓"
            else:
                grid[cy][cx] = "·"
        # Patrol
        px, py = self._patrol_pos
        if 0 <= px < self._WIDTH and 0 <= py < self._HEIGHT:
            grid[py][px] = "e"
        # Player
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "open space",
            "·": "unpainted track",
            "▓": "painted track",
            "e": "patrol enemy",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"you (facing {self._DIR_NAMES.get(self._player_dir, 'down')})"

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Painted: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}\n"
            f"You: ({self._player_x},{self._player_y})    "
            f"Patrol: ({px},{py})"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Amidar on a 12x10 grid. A 3x4 rectangle (corners 3,3 to "
            "5,6) has 10 unpainted track cells (·) on its perimeter. "
            "Stepping onto a track cell paints it (▓). A patrol enemy (e) "
            "walks the perimeter clockwise, advancing 1 cell every 3 "
            "ticks. LEFT/RIGHT/UP/DOWN moves you 1 cell. Reward: +1/10 per "
            "newly painted cell. If the patrol's cell coincides with you, "
            "you take a -1 terminal penalty."
        )
