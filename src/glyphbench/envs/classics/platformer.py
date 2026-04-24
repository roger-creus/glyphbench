"""Platformer: side-scrolling Mario-style platformer.

Gym IDs:
  glyphbench/classics-platformer-v0
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

PLATFORMER_ACTION_SPEC = ActionSpec(
    names=("LEFT", "RIGHT", "JUMP", "NOOP"),
    descriptions=(
        "move 1 cell left",
        "move 1 cell right",
        "jump up 3 cells (only if grounded)",
        "do nothing this turn",
    ),
)

VIEW_W = 20
VIEW_H = 12
LEVEL_W = 80  # total level width
LEVEL_H = 12
GROUND_Y = LEVEL_H - 1  # bottom row is ground (y=11)
FLAG_MIN_X = 60

SYM_PLAYER = "@"
SYM_PLATFORM = "\u25ac"  # ▬
SYM_ENEMY = "E"
SYM_FLAG = "\u2691"       # ⚑
SYM_AIR = "\u00b7"        # ·
SYM_GROUND = "\u2588"     # █

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class PlatformerEnv(BaseAsciiEnv):
    """Side-scrolling Mario-style platformer."""

    action_spec = PLATFORMER_ACTION_SPEC
    noop_action_name: str = "NOOP"

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._px: int = 0
        self._py: int = 0
        self._vy: int = 0  # vertical velocity (positive = falling)
        self._grounded: bool = False
        self._max_x_reached: int = 0
        self._platforms: list[tuple[int, int, int]] = []  # (x_start, x_end, y)
        self._enemies: list[list[int]] = []  # [[x, y, dx], ...] mutable
        self._flag_x: int = 0
        self._flag_y: int = 0
        self._ground_gaps: set[int] = set()  # x positions with no ground
        self._message: str = ""

    def env_id(self) -> str:
        return "glyphbench/classics-platformer-v0"

    # ------------------------------------------------------------------
    # Level generation
    # ------------------------------------------------------------------

    def _generate_level(self) -> None:
        """Procedurally generate platforms, enemies, gaps, and flag."""
        # Ground gaps (pits) - avoid first 5 columns
        self._ground_gaps = set()
        x = 8
        while x < LEVEL_W - 5:
            if self.rng.random() < 0.25:
                gap_len = int(self.rng.integers(2, 4))
                for gx in range(x, min(x + gap_len, LEVEL_W - 5)):
                    self._ground_gaps.add(gx)
                x += gap_len + 3
            else:
                x += 1

        # Platforms at various heights
        self._platforms = []
        x = 5
        while x < LEVEL_W - 5:
            if self.rng.random() < 0.4:
                plat_len = int(self.rng.integers(3, 7))
                plat_y = int(self.rng.integers(4, GROUND_Y - 1))
                self._platforms.append((x, x + plat_len - 1, plat_y))
                x += plat_len + 2
            else:
                x += 1

        # Ensure platforms over gaps for reachability
        for gx in sorted(self._ground_gaps):
            # Check if there's already a platform near this gap
            has_platform = False
            for px_start, px_end, py in self._platforms:
                if px_start <= gx <= px_end:
                    has_platform = True
                    break
            if not has_platform and self.rng.random() < 0.6:
                plat_y = int(self.rng.integers(GROUND_Y - 3, GROUND_Y))
                self._platforms.append((gx - 1, gx + 2, plat_y))

        # Enemies on platforms and ground
        self._enemies = []
        # Ground enemies
        for _ in range(int(self.rng.integers(3, 7))):
            ex = int(self.rng.integers(10, LEVEL_W - 10))
            if ex not in self._ground_gaps:
                self._enemies.append([ex, GROUND_Y - 1, int(self.rng.choice([-1, 1]))])

        # Platform enemies
        for px_start, px_end, py in self._platforms:
            if px_end - px_start >= 3 and self.rng.random() < 0.3:
                ex = int(self.rng.integers(px_start, px_end + 1))
                self._enemies.append([ex, py - 1, int(self.rng.choice([-1, 1]))])

        # Flag at the end
        self._flag_x = int(self.rng.integers(FLAG_MIN_X, LEVEL_W - 2))
        self._flag_y = GROUND_Y - 1
        # Make sure flag is not over a gap
        while self._flag_x in self._ground_gaps:
            self._flag_x += 1

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------

    def _is_solid(self, x: int, y: int) -> bool:
        """Check if a cell is solid (ground or platform)."""
        if y < 0 or x < 0 or x >= LEVEL_W:
            return False
        if y >= LEVEL_H:
            return False
        # Ground
        if y == GROUND_Y and x not in self._ground_gaps:
            return True
        # Platforms
        for px_start, px_end, py in self._platforms:
            if py == y and px_start <= x <= px_end:
                return True
        return False

    def _check_grounded(self) -> bool:
        """Check if player is standing on solid ground."""
        return self._is_solid(self._px, self._py + 1)

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._px = 2
        self._py = GROUND_Y - 1  # stand on ground
        self._vy = 0
        self._max_x_reached = 2
        self._message = "Reach the flag!"
        self._generate_level()
        self._grounded = self._check_grounded()
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        reward = 0.0
        terminated = False
        self._message = ""

        name = self.action_spec.names[action]

        # --- Horizontal movement ---
        new_px = self._px
        if name == "LEFT":
            new_px = self._px - 1
        elif name == "RIGHT":
            new_px = self._px + 1
        elif name == "JUMP":
            if self._grounded:
                self._vy = -3  # jump velocity (negative = up)

        # Clamp horizontal
        if 0 <= new_px < LEVEL_W and not self._is_solid(new_px, self._py):
            self._px = new_px

        # --- Apply gravity ---
        if not self._grounded or self._vy < 0:
            # Apply velocity
            if self._vy < 0:
                # Moving upward
                for _ in range(-self._vy):
                    if self._py - 1 >= 0 and not self._is_solid(self._px, self._py - 1):
                        self._py -= 1
                    else:
                        self._vy = 0
                        break
                self._vy += 1  # gravity reduces upward velocity
            else:
                # Falling
                if not self._is_solid(self._px, self._py + 1):
                    self._py += 1
                    self._vy = min(self._vy + 1, 3)  # terminal velocity
                else:
                    self._vy = 0
        else:
            # On ground, apply gravity check in case ground disappeared
            if not self._is_solid(self._px, self._py + 1):
                self._vy = 1
                self._py += 1

        self._grounded = self._check_grounded()

        # --- Check fall off bottom ---
        if self._py >= LEVEL_H:
            terminated = True
            reward = -1.0
            self._message = "Fell into a pit! Game over."
            info["outcome"] = "fell"
            return self._render_current_observation(), reward, terminated, False, info

        # --- Move enemies ---
        for enemy in self._enemies:
            ex, ey, edx = enemy
            new_ex = ex + edx
            # Reverse if hitting edge or gap or no ground
            on_ground_next = self._is_solid(new_ex, ey + 1)
            blocked = self._is_solid(new_ex, ey)
            if new_ex < 0 or new_ex >= LEVEL_W or not on_ground_next or blocked:
                enemy[2] = -edx  # reverse
                new_ex = ex + enemy[2]
                if not (0 <= new_ex < LEVEL_W and self._is_solid(new_ex, ey + 1)
                        and not self._is_solid(new_ex, ey)):
                    new_ex = ex  # stay put
            enemy[0] = new_ex

        # --- Check enemy collision ---
        for enemy in self._enemies:
            if enemy[0] == self._px and enemy[1] == self._py:
                terminated = True
                reward = -1.0
                self._message = "Hit by an enemy! Game over."
                info["outcome"] = "enemy"
                return self._render_current_observation(), reward, terminated, False, info

        # --- Check flag ---
        if self._px == self._flag_x and self._py == self._flag_y:
            terminated = True
            reward = 10.0
            self._message = "Reached the flag! Victory!"
            info["outcome"] = "victory"
            return self._render_current_observation(), reward, terminated, False, info

        # --- Progress reward ---
        if self._px > self._max_x_reached:
            reward += 0.1 * (self._px - self._max_x_reached)
            self._max_x_reached = self._px

        info["player_x"] = self._px
        info["max_x"] = self._max_x_reached
        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        # Compute viewport: center on player horizontally
        vp_left = max(0, self._px - VIEW_W // 2)
        vp_left = min(vp_left, max(0, LEVEL_W - VIEW_W))
        vp_right = vp_left + VIEW_W

        grid = make_empty_grid(VIEW_W, VIEW_H, SYM_AIR)

        # Draw ground
        for x in range(vp_left, vp_right):
            lx = x - vp_left
            if x not in self._ground_gaps:
                grid[GROUND_Y][lx] = SYM_GROUND

        # Draw platforms
        for px_start, px_end, py in self._platforms:
            for x in range(max(px_start, vp_left), min(px_end + 1, vp_right)):
                lx = x - vp_left
                if 0 <= py < LEVEL_H:
                    grid[py][lx] = SYM_PLATFORM

        # Draw flag
        if vp_left <= self._flag_x < vp_right:
            fx = self._flag_x - vp_left
            if 0 <= self._flag_y < LEVEL_H:
                grid[self._flag_y][fx] = SYM_FLAG

        # Draw enemies
        for enemy in self._enemies:
            ex, ey, _ = enemy
            if vp_left <= ex < vp_right and 0 <= ey < LEVEL_H:
                grid[ey][ex - vp_left] = SYM_ENEMY

        # Draw player (clamped to visible)
        if 0 <= self._py < LEVEL_H:
            player_vx = self._px - vp_left
            if 0 <= player_vx < VIEW_W:
                grid[self._py][player_vx] = SYM_PLAYER

        legend = build_legend({
            SYM_PLAYER: "player",
            SYM_PLATFORM: "platform",
            SYM_ENEMY: "enemy",
            SYM_FLAG: "flag (goal)",
            SYM_AIR: "air",
            SYM_GROUND: "ground",
        })

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Position: ({self._px}, {self._py})    "
            f"Grounded: {'yes' if self._grounded else 'no'}    "
            f"Progress: {self._px}/{self._flag_x}"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=legend,
            hud=hud,
            message=self._message,
        )

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            "TASK\n"
            "Navigate a side-scrolling platformer. Jump across platforms, avoid enemies, "
            "and reach the flag at the far right of the level.\n\n"
            "RULES\n"
            f"- The viewport is {VIEW_W} wide x {VIEW_H} tall and scrolls to follow you.\n"
            f"- The full level is {LEVEL_W} cells wide.\n"
            "- LEFT/RIGHT move 1 cell horizontally.\n"
            "- JUMP launches you upward 3 cells (only works when grounded).\n"
            "- Gravity pulls you down 1 cell per step when airborne.\n"
            "- Falling off the bottom of the level ends the game (-1 reward).\n"
            "- Touching an enemy ends the game (-1 reward).\n"
            "- Reaching the flag wins (+10 reward).\n"
            "- +0.1 reward for each new rightward position reached.\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_env(
    "glyphbench/classics-platformer-v0",
    "glyphbench.envs.classics.platformer:PlatformerEnv",
)
