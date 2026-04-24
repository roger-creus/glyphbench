"""Pokemon-style ice sliding puzzle.

Player slides in a direction until hitting a wall or rock. Reach the goal.

Gym IDs:
  glyphbench/classics-icesliding-easy-v0    (8x8)
  glyphbench/classics-icesliding-medium-v0  (10x10)
  glyphbench/classics-icesliding-hard-v0    (12x12)
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

ICE_ACTION_SPEC = ActionSpec(
    names=("UP", "DOWN", "LEFT", "RIGHT"),
    descriptions=(
        "slide upward until hitting a wall or rock",
        "slide downward until hitting a wall or rock",
        "slide left until hitting a wall or rock",
        "slide right until hitting a wall or rock",
    ),
)

SYM_PLAYER = "@"
SYM_ICE = "\u2591"     # ░
SYM_ROCK = "\u25c6"    # ◆
SYM_GOAL = "\u2605"    # ★
SYM_WALL = "\u2588"    # █

_DIR_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

DIFFICULTY = {
    "easy": {"size": 8, "rocks": 6, "max_turns": 100},
    "medium": {"size": 10, "rocks": 10, "max_turns": 150},
    "hard": {"size": 12, "rocks": 15, "max_turns": 200},
}

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class _IceSlidingBase(BaseAsciiEnv):
    """Pokemon-style ice puzzle: slide until hitting wall/rock."""

    action_spec = ICE_ACTION_SPEC
    noop_action_name: str = "UP"

    _grid_size: int = 8
    _num_rocks: int = 6
    _difficulty: str = "easy"

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)
        self._player: tuple[int, int] = (0, 0)
        self._goal: tuple[int, int] = (0, 0)
        self._rocks: set[tuple[int, int]] = set()

    def env_id(self) -> str:
        return f"glyphbench/classics-icesliding-{self._difficulty}-v0"

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _is_wall(self, x: int, y: int) -> bool:
        s = self._grid_size
        return x <= 0 or x >= s - 1 or y <= 0 or y >= s - 1

    def _blocked(self, x: int, y: int) -> bool:
        return self._is_wall(x, y) or (x, y) in self._rocks

    def _slide_result(self, sx: int, sy: int, dx: int, dy: int) -> tuple[int, int]:
        """Return where a slide from (sx, sy) in direction (dx, dy) ends up."""
        cx, cy = sx, sy
        while True:
            nx, ny = cx + dx, cy + dy
            if self._blocked(nx, ny):
                return (cx, cy)
            cx, cy = nx, ny

    def _reachable_bfs(self, start: tuple[int, int], goal: tuple[int, int]) -> bool:
        """BFS over ice-slide transitions to check reachability."""
        visited: set[tuple[int, int]] = {start}
        queue: deque[tuple[int, int]] = deque([start])
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) == goal:
                return True
            for dx, dy in _DIR_DELTAS.values():
                nx, ny = self._slide_result(cx, cy, dx, dy)
                if (nx, ny) != (cx, cy) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False

    def _generate_puzzle(self) -> None:
        """Generate a solvable ice-sliding puzzle."""
        s = self._grid_size
        interior = [(x, y) for x in range(1, s - 1) for y in range(1, s - 1)]

        for _ in range(500):
            self.rng.shuffle(interior)  # type: ignore[arg-type]
            self._player = interior[0]
            self._goal = interior[1]
            self._rocks = set()
            for i in range(2, 2 + self._num_rocks):
                if i < len(interior):
                    self._rocks.add(interior[i])
            if self._reachable_bfs(self._player, self._goal):
                return
        # Fallback: no rocks, player and goal on same row
        self._rocks = set()
        self._player = (1, 1)
        self._goal = (s - 2, 1)

    # ------------------------------------------------------------------
    # Core loop
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
        nx, ny = self._slide_result(px, py, dx, dy)
        self._player = (nx, ny)

        reward = -0.01
        terminated = False

        if self._player == self._goal:
            reward = 1.0
            terminated = True
            info["goal_reached"] = True

        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        s = self._grid_size
        grid = make_empty_grid(s, s, SYM_ICE)

        # Walls (border)
        for x in range(s):
            grid[0][x] = SYM_WALL
            grid[s - 1][x] = SYM_WALL
        for y in range(s):
            grid[y][0] = SYM_WALL
            grid[y][s - 1] = SYM_WALL

        # Rocks
        for rx, ry in self._rocks:
            grid[ry][rx] = SYM_ROCK

        # Goal
        gx, gy = self._goal
        grid[gy][gx] = SYM_GOAL

        # Player
        px, py = self._player
        grid[py][px] = SYM_PLAYER

        legend = build_legend({
            SYM_PLAYER: "you",
            SYM_ICE: "ice (you slide over this)",
            SYM_ROCK: "rock (stops sliding)",
            SYM_GOAL: "goal",
            SYM_WALL: "wall",
        })

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Position: ({px}, {py})"
        )

        return GridObservation(grid=grid_to_string(grid), legend=legend, hud=hud, message="")

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Navigate an ice-sliding puzzle. When you move in a direction, you slide "
            "continuously until you hit a wall or a rock. Reach the goal.\n\n"
            "RULES\n"
            f"- The grid is {self._grid_size}x{self._grid_size} with walls around the border.\n"
            "- Moving on ice causes you to slide until you hit a wall or rock.\n"
            "- Rocks stop your sliding but you cannot pass through them.\n"
            "- Reach the goal to win (+1 reward). Each move costs -0.01.\n"
            "- Plan your path carefully: you cannot stop on ice mid-slide.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class IceSlidingEasyEnv(_IceSlidingBase):
    _grid_size = 8
    _num_rocks = 6
    _difficulty = "easy"

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)


class IceSlidingMediumEnv(_IceSlidingBase):
    _grid_size = 10
    _num_rocks = 10
    _difficulty = "medium"

    def __init__(self, max_turns: int = 150) -> None:
        super().__init__(max_turns=max_turns)


class IceSlidingHardEnv(_IceSlidingBase):
    _grid_size = 12
    _num_rocks = 15
    _difficulty = "hard"

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_env(
    "glyphbench/classics-icesliding-easy-v0",
    "glyphbench.envs.classics.ice_sliding:IceSlidingEasyEnv",
)
register_env(
    "glyphbench/classics-icesliding-medium-v0",
    "glyphbench.envs.classics.ice_sliding:IceSlidingMediumEnv",
)
register_env(
    "glyphbench/classics-icesliding-hard-v0",
    "glyphbench.envs.classics.ice_sliding:IceSlidingHardEnv",
)
