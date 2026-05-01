"""miniatari Kangaroo.

Identity: Climb 3 floors of a tower while avoiding a falling hazard.
Win condition: reach the top floor.
Reward: Pattern D, +1/3 per floor reached; -1 on falling hazard hit.

Gym ID: glyphbench/miniatari-kangaroo-v0

Random baseline (seed=0..29): success_rate=0%, mean_length=300, mean_return=+0.222
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniKangarooEnv(MiniatariBase):
    """Mini Kangaroo: 12x14 tower with 3 floors and a hazard.

    The tower has 4 floors (rows 12, 9, 6, 3). Floor 0 is start (row 12),
    Floor 1 (row 9), Floor 2 (row 6), Floor 3 (row 3, top = goal).
    Each floor is connected to the floor above by a ladder column. Ladders
    alternate sides: floor 0->1 ladder at col 1; floor 1->2 ladder at col
    10; floor 2->3 ladder at col 1. Reaching a higher floor for the first
    time gives +1/3. A coconut hazard (o) drops every 4 ticks from row
    0 in a column chosen each drop; if it lands on the player's cell,
    -1 terminal. Hazards disappear when they hit a floor.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN"),
        descriptions=(
            "do nothing",
            "move left along the floor",
            "move right along the floor",
            "climb up the ladder you stand on",
            "climb down the ladder you stand on",
        ),
    )

    default_max_turns = 300

    _WIDTH = 12
    _HEIGHT = 14
    _N_FLOORS = 3  # number of upward transitions to win
    _WIN_TARGET = _N_FLOORS
    _FLOOR_ROWS = (12, 9, 6, 3)  # floor index 0..3
    _LADDER_COLS = (1, 10, 1)  # ladder col for transition i -> i+1
    _COCONUT_DROP_EVERY = 4

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._floor: int = 0
        self._max_floor: int = 0
        # coconuts: [x, y]; fall down by 1 each tick
        self._coconuts: list[list[int]] = []
        self._tick_count: int = 0
        self._progress: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-kangaroo-v0"

    def _on_floor(self, y: int) -> int:
        for i, fy in enumerate(self._FLOOR_ROWS):
            if y == fy:
                return i
        return -1

    def _on_ladder(self, x: int, y: int) -> int:
        """Return ladder index (0..N-1) if (x,y) is between floor i and i+1."""
        for i, lcol in enumerate(self._LADDER_COLS):
            top = self._FLOOR_ROWS[i + 1]
            bot = self._FLOOR_ROWS[i]
            if x == lcol and top <= y <= bot:
                return i
        return -1

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._tick_count = 0
        self._progress = 0
        self._floor = 0
        self._max_floor = 0
        self._coconuts = []
        # Player on bottom floor near the ladder up
        self._player_x = self._LADDER_COLS[0] + 1
        self._player_y = self._FLOOR_ROWS[0]
        self._player_dir = (1, 0)

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1

        nx, ny = self._player_x, self._player_y
        if action_name == "LEFT":
            nx -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            nx += 1
            self._player_dir = (1, 0)
        elif action_name == "UP":
            ny -= 1
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            ny += 1
            self._player_dir = (0, 1)

        # Movement validation:
        if action_name in ("LEFT", "RIGHT"):
            # Must stay on a floor row
            if 0 <= nx < self._WIDTH and self._on_floor(ny) >= 0:
                self._player_x, self._player_y = nx, ny
        elif action_name == "UP":
            # Must be on a ladder cell (or transitioning from floor onto ladder going up)
            li = self._on_ladder(self._player_x, ny)
            if li >= 0 or self._on_ladder(self._player_x, self._player_y) >= 0:
                if 0 <= ny < self._HEIGHT and self._player_x in self._LADDER_COLS:
                    self._player_y = ny
        elif action_name == "DOWN":
            li = self._on_ladder(self._player_x, ny)
            if li >= 0 or self._on_ladder(self._player_x, self._player_y) >= 0:
                if 0 <= ny < self._HEIGHT and self._player_x in self._LADDER_COLS:
                    self._player_y = ny

        # Detect floor reached
        cur_floor = self._on_floor(self._player_y)
        if cur_floor >= 0:
            self._floor = cur_floor
            if cur_floor > self._max_floor:
                # award +1/3 per new floor reached
                gained = cur_floor - self._max_floor
                reward += gained * self._progress_reward(self._WIN_TARGET)
                self._max_floor = cur_floor
                self._progress = cur_floor
                self._message = f"Reached floor {cur_floor}! ({self._progress}/{self._WIN_TARGET})"
                if self._max_floor >= self._N_FLOORS:
                    self._on_won()
                    return reward, self._game_over, info

        # Coconut drop
        rng = self.rng
        if self._tick_count % self._COCONUT_DROP_EVERY == 0:
            cx = int(rng.integers(0, self._WIDTH))
            self._coconuts.append([cx, 0])

        # Coconut fall: 1 cell per tick
        new_coconuts: list[list[int]] = []
        for cx, cy in self._coconuts:
            ncy = cy + 1
            if ncy >= self._HEIGHT:
                continue  # off screen
            # If lands on a floor row, stop (disappears)
            if self._on_floor(ncy) >= 0 and ncy != self._FLOOR_ROWS[0]:
                # hits floor, don't carry over
                # but check if it caught the player on the way
                if cx == self._player_x and ncy == self._player_y:
                    self._message = "Hit by a coconut!"
                    reward += self._death_reward()
                    self._on_life_lost()
                    return reward, True, info
                continue
            # Player hit?
            if cx == self._player_x and ncy == self._player_y:
                self._message = "Hit by a coconut!"
                reward += self._death_reward()
                self._on_life_lost()
                return reward, True, info
            new_coconuts.append([cx, ncy])
        self._coconuts = new_coconuts

        info["progress"] = self._progress
        info["floor"] = self._floor
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Floors
        for fy in self._FLOOR_ROWS:
            for x in range(self._WIDTH):
                grid[fy][x] = "─"
        # Ladders
        for i, lcol in enumerate(self._LADDER_COLS):
            top = self._FLOOR_ROWS[i + 1]
            bot = self._FLOOR_ROWS[i]
            for y in range(top, bot + 1):
                grid[y][lcol] = "║"
        # Coconuts
        for cx, cy in self._coconuts:
            if 0 <= cx < self._WIDTH and 0 <= cy < self._HEIGHT:
                grid[cy][cx] = "o"
        # Player
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            pch = self._DIR_CHARS.get(self._player_dir, "@")
            grid[self._player_y][self._player_x] = pch

        symbols = {
            " ": "open air",
            "─": "floor",
            "║": "ladder",
            "o": "falling coconut",
        }
        pch = self._DIR_CHARS.get(self._player_dir, "@")
        symbols[pch] = f"you (facing {self._DIR_NAMES.get(self._player_dir, 'right')})"

        coconuts_info = " ".join(f"({cx},{cy})" for cx, cy in self._coconuts)
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Floor: {self._floor} (max {self._max_floor}/{self._N_FLOORS})    "
            f"Score: {self._score:.3f}\n"
            f"You: ({self._player_x},{self._player_y})    "
            f"Coconuts: {coconuts_info}"
        )
        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Kangaroo on a 12x14 tower. 4 floors at rows 12, 9, 6, 3 "
            "(─), connected by ladders (║) at columns 1, 10, 1 (alternating). "
            "Start on floor 0 (bottom). LEFT/RIGHT moves you along a floor; "
            "UP/DOWN climbs up/down the ladder under you. Coconuts (o) drop "
            "from the ceiling every 4 ticks at random columns and fall 1 "
            "row per tick — being hit is -1 terminal. Reaching a higher "
            "floor for the first time grants +1/3. Reach floor 3 to win."
        )
