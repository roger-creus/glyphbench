"""Gravity maze: navigate a ball through platforms with gravity.

The ball falls unless standing on a platform. Move left/right or jump.

Gym IDs:
  glyphbench/classics-gravitymaze-v0  (12x8)
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

GRAVITY_ACTION_SPEC = ActionSpec(
    names=("LEFT", "RIGHT", "JUMP"),
    descriptions=(
        "move the ball one cell to the left",
        "move the ball one cell to the right",
        "jump up 2 cells (only works if grounded)",
    ),
)

WIDTH = 12
HEIGHT = 8

SYM_BALL = "\u25cf"      # ●
SYM_PLATFORM = "\u25ac"  # ▬
SYM_GOAL = "\u2605"      # ★
SYM_AIR = "\u00b7"       # ·
SYM_WALL = "\u2588"      # █

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class GravityMazeEnv(BaseGlyphEnv):
    """Navigate a ball through a gravity maze to reach the goal."""

    action_spec = GRAVITY_ACTION_SPEC
    noop_action_name: str = "LEFT"

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._ball: tuple[int, int] = (0, 0)  # (x, y)
        self._goal: tuple[int, int] = (0, 0)
        self._platforms: set[tuple[int, int]] = set()

    def env_id(self) -> str:
        return "glyphbench/classics-gravitymaze-v0"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_solid(self, x: int, y: int) -> bool:
        """Check if a cell is solid (wall or platform)."""
        if x <= 0 or x >= WIDTH - 1 or y <= 0 or y >= HEIGHT - 1:
            return True  # walls
        return (x, y) in self._platforms

    def _is_grounded(self) -> bool:
        """Check if ball is standing on a solid surface."""
        bx, by = self._ball
        return self._is_solid(bx, by + 1)

    def _apply_gravity(self) -> None:
        """Apply gravity: ball falls if not grounded."""
        bx, by = self._ball
        while not self._is_solid(bx, by + 1):
            by += 1
            if by >= HEIGHT - 1:
                break
        self._ball = (bx, by)

    def _fell_off(self) -> bool:
        """Check if ball is at the bottom wall (fell off)."""
        return self._ball[1] >= HEIGHT - 2 and not self._is_grounded()

    def _generate_maze(self) -> None:
        """Procedurally generate platforms and place goal."""
        self._platforms = set()

        # Create a ground floor
        for x in range(1, WIDTH - 1):
            self._platforms.add((x, HEIGHT - 2))

        # Create platforms at various heights
        num_platforms = int(self.rng.integers(4, 8))
        for _ in range(num_platforms):
            py = int(self.rng.integers(2, HEIGHT - 3))
            px_start = int(self.rng.integers(1, WIDTH - 4))
            length = int(self.rng.integers(2, 5))
            for x in range(px_start, min(px_start + length, WIDTH - 1)):
                self._platforms.add((x, py))

        # Remove some ground floor sections to create gaps
        n_gaps = int(self.rng.integers(1, 4))
        for _ in range(n_gaps):
            gx = int(self.rng.integers(3, WIDTH - 3))
            gap_len = int(self.rng.integers(1, 3))
            for x in range(gx, min(gx + gap_len, WIDTH - 1)):
                self._platforms.discard((x, HEIGHT - 2))

        # Place ball on a platform
        ground_cells = [(x, HEIGHT - 3) for x in range(1, WIDTH - 1)
                        if (x, HEIGHT - 2) in self._platforms]
        if not ground_cells:
            ground_cells = [(1, HEIGHT - 3)]
        ball_cell = ground_cells[int(self.rng.integers(0, len(ground_cells)))]
        self._ball = ball_cell

        # Place goal on a platform (preferably elevated)
        platform_tops: list[tuple[int, int]] = []
        for px, py in self._platforms:
            # Cell above platform must be empty and not a wall
            if py > 1 and (px, py - 1) not in self._platforms:
                platform_tops.append((px, py - 1))

        # Filter out ball position
        platform_tops = [p for p in platform_tops if p != self._ball]

        if platform_tops:
            idx = int(self.rng.integers(0, len(platform_tops)))
            self._goal = platform_tops[idx]
        else:
            self._goal = (WIDTH - 2, HEIGHT - 3)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._generate_maze()
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        name = self.action_spec.names[action]
        bx, by = self._ball

        if name == "LEFT":
            nx = bx - 1
            if not self._is_solid(nx, by):
                self._ball = (nx, by)
        elif name == "RIGHT":
            nx = bx + 1
            if not self._is_solid(nx, by):
                self._ball = (nx, by)
        elif name == "JUMP":
            if self._is_grounded():
                # Jump up 2 cells
                ny = by - 1
                if not self._is_solid(bx, ny):
                    ny2 = by - 2
                    if not self._is_solid(bx, ny2):
                        self._ball = (bx, ny2)
                    else:
                        self._ball = (bx, ny)

        # Apply gravity after horizontal movement
        self._apply_gravity()

        reward = 0.0
        terminated = False

        # Check if at bottom wall row (fell into a gap)
        if self._ball[1] >= HEIGHT - 2 and (self._ball[0], HEIGHT - 2) not in self._platforms:
            # Ball is at wall level = fell off
            reward = -1.0
            terminated = True
            info["fell"] = True
        elif self._ball == self._goal:
            reward = 1.0
            terminated = True
            info["goal_reached"] = True

        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(WIDTH, HEIGHT, SYM_AIR)

        # Walls (border)
        for x in range(WIDTH):
            grid[0][x] = SYM_WALL
            grid[HEIGHT - 1][x] = SYM_WALL
        for y in range(HEIGHT):
            grid[y][0] = SYM_WALL
            grid[y][WIDTH - 1] = SYM_WALL

        # Platforms
        for px, py in self._platforms:
            if 0 < px < WIDTH - 1 and 0 < py < HEIGHT - 1:
                grid[py][px] = SYM_PLATFORM

        # Goal
        gx, gy = self._goal
        grid[gy][gx] = SYM_GOAL

        # Ball
        bx, by = self._ball
        if 0 < bx < WIDTH - 1 and 0 < by < HEIGHT - 1:
            grid[by][bx] = SYM_BALL

        legend = build_legend({
            SYM_BALL: "ball (you)",
            SYM_PLATFORM: "platform",
            SYM_GOAL: "goal",
            SYM_AIR: "air",
            SYM_WALL: "wall",
        })

        grounded = self._is_grounded()
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Position: ({bx}, {by})    "
            f"Grounded: {'yes' if grounded else 'no'}"
        )

        return GridObservation(grid=grid_to_string(grid), legend=legend, hud=hud, message="")

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Navigate a ball through a gravity maze to reach the goal.\n\n"
            "RULES\n"
            f"- The grid is {WIDTH}x{HEIGHT} (width x height).\n"
            "- Gravity pulls the ball down each step. The ball falls until it lands on a platform.\n"
            "- LEFT/RIGHT move the ball one cell horizontally. Gravity applies after.\n"
            "- JUMP moves the ball up 2 cells (only works when grounded on a platform).\n"
            "- Reaching the goal gives +1 reward.\n"
            "- Falling off the bottom gives -1 reward.\n"
            "- Plan jumps carefully to reach elevated platforms.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

