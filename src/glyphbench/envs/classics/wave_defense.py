"""Wave Defense: central rotating cannon defends against enemy waves.

Gym IDs:
  glyphbench/classics-wavedefense-v0
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

WAVE_DEFENSE_ACTION_SPEC = ActionSpec(
    names=("ROTATE_CW", "ROTATE_CCW", "FIRE", "WAIT"),
    descriptions=(
        "rotate cannon clockwise",
        "rotate cannon counter-clockwise",
        "fire a projectile in the current facing direction",
        "do nothing this turn",
    ),
)

GRID_SIZE = 15
CENTER = 7  # center of 15x15 grid (0-indexed)
TOTAL_WAVES = 5

# 8 compass directions in clockwise order
DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

# (dx, dy) deltas for each direction (y increases downward)
DIR_DELTAS: dict[str, tuple[int, int]] = {
    "N":  (0, -1),
    "NE": (1, -1),
    "E":  (1, 0),
    "SE": (1, 1),
    "S":  (0, 1),
    "SW": (-1, 1),
    "W":  (-1, 0),
    "NW": (-1, -1),
}

DIR_ARROWS: dict[str, str] = {
    "N": "\u2191",   # ↑
    "NE": "\u2197",  # ↗
    "E": "\u2192",   # →
    "SE": "\u2198",  # ↘
    "S": "\u2193",   # ↓
    "SW": "\u2199",  # ↙
    "W": "\u2190",   # ←
    "NW": "\u2196",  # ↖
}

# Bullet trail characters based on direction
TRAIL_CHARS: dict[str, str] = {
    "N":  "\u2502",  # │
    "S":  "\u2502",  # │
    "E":  "\u2500",  # ─
    "W":  "\u2500",  # ─
    "NE": "\u2571",  # ╱
    "SW": "\u2571",  # ╱
    "NW": "\u2572",  # ╲
    "SE": "\u2572",  # ╲
}

SYM_CANNON = "\u2295"  # ⊕
SYM_ENEMY = "E"
SYM_FLOOR = "\u00b7"   # ·
SYM_BORDER = "\u2588"  # █

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class WaveDefenseEnv(BaseGlyphEnv):
    """Central rotating cannon defends against waves of enemies."""

    action_spec = WAVE_DEFENSE_ACTION_SPEC
    noop_action_name: str = "WAIT"

    def __init__(self, max_turns: int = 500) -> None:
        super().__init__(max_turns=max_turns)
        self._facing_idx: int = 0  # index into DIRECTIONS
        self._enemies: list[tuple[int, int]] = []
        self._current_wave: int = 0
        self._enemies_remaining_in_wave: int = 0
        self._steps_since_last_spawn: int = 0
        self._enemies_spawned_this_wave: int = 0
        self._kills: int = 0
        self._waves_cleared: int = 0
        self._trail: list[tuple[int, int]] = []  # bullet trail cells (visual, 1 step)
        self._trail_dir: str = "N"
        self._all_waves_done: bool = False
        self._message: str = ""

    def env_id(self) -> str:
        return "glyphbench/classics-wavedefense-v0"

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._facing_idx = 0  # start facing North
        self._enemies = []
        self._current_wave = 0
        self._enemies_remaining_in_wave = 0
        self._enemies_spawned_this_wave = 0
        self._steps_since_last_spawn = 0
        self._kills = 0
        self._waves_cleared = 0
        self._trail = []
        self._trail_dir = "N"
        self._all_waves_done = False
        self._message = "Wave 1 begins!"
        self._start_wave(0)
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        reward = 0.0
        terminated = False
        self._trail = []
        self._message = ""

        name = self.action_spec.names[action]

        # --- Process player action ---
        if name == "ROTATE_CW":
            self._facing_idx = (self._facing_idx + 1) % 8
        elif name == "ROTATE_CCW":
            self._facing_idx = (self._facing_idx - 1) % 8
        elif name == "FIRE":
            kill_reward = self._fire()
            reward += kill_reward
        # WAIT does nothing

        # --- Spawn enemies ---
        self._steps_since_last_spawn += 1
        if (self._steps_since_last_spawn >= 3
                and self._enemies_spawned_this_wave < self._wave_enemy_count(self._current_wave)):
            self._spawn_enemy()
            self._steps_since_last_spawn = 0

        # --- Move enemies toward center ---
        new_enemies: list[tuple[int, int]] = []
        for ex, ey in self._enemies:
            nx, ny = self._move_toward_center(ex, ey)
            if nx == CENTER and ny == CENTER:
                # Enemy reached center
                reward -= 1.0
                terminated = True
                self._message = "An enemy reached the center! Game over."
                info["outcome"] = "enemy_reached_center"
                return self._render_current_observation(), reward, terminated, False, info
            new_enemies.append((nx, ny))
        self._enemies = new_enemies

        # --- Check wave completion ---
        if (not self._all_waves_done
                and self._enemies_spawned_this_wave >= self._wave_enemy_count(self._current_wave)
                and len(self._enemies) == 0):
            # Wave cleared
            reward += 5.0
            self._waves_cleared += 1
            self._message = f"Wave {self._current_wave + 1} cleared!"
            if self._current_wave + 1 >= TOTAL_WAVES:
                # All waves done
                reward += 1.0
                terminated = True
                self._all_waves_done = True
                self._message = "All waves cleared! Victory!"
                info["outcome"] = "victory"
            else:
                self._current_wave += 1
                self._start_wave(self._current_wave)
                self._message += f" Wave {self._current_wave + 1} begins!"

        info["kills"] = self._kills
        info["wave"] = self._current_wave + 1
        info["waves_cleared"] = self._waves_cleared
        return self._render_current_observation(), reward, terminated, False, info

    # ------------------------------------------------------------------
    # Wave / spawn helpers
    # ------------------------------------------------------------------

    def _wave_enemy_count(self, wave: int) -> int:
        """Wave N has N+2 enemies."""
        return wave + 2

    def _start_wave(self, wave: int) -> None:
        self._enemies_spawned_this_wave = 0
        self._steps_since_last_spawn = 2  # spawn on first step of wave

    def _spawn_enemy(self) -> None:
        """Spawn an enemy at a random edge cell."""
        edge_cells: list[tuple[int, int]] = []
        # Top and bottom rows (inside border)
        for x in range(1, GRID_SIZE - 1):
            edge_cells.append((x, 1))
            edge_cells.append((x, GRID_SIZE - 2))
        # Left and right columns (inside border)
        for y in range(2, GRID_SIZE - 2):
            edge_cells.append((1, y))
            edge_cells.append((GRID_SIZE - 2, y))

        # Avoid spawning on top of existing enemies
        occupied = set(self._enemies)
        available = [c for c in edge_cells if c not in occupied]
        if not available:
            available = edge_cells  # fallback

        idx = int(self.rng.integers(0, len(available)))
        self._enemies.append(available[idx])
        self._enemies_spawned_this_wave += 1

    def _move_toward_center(self, ex: int, ey: int) -> tuple[int, int]:
        """Move enemy 1 cell toward center."""
        dx = 0 if ex == CENTER else (1 if ex < CENTER else -1)
        dy = 0 if ey == CENTER else (1 if ey < CENTER else -1)
        # Move in the axis with greater distance, or both if diagonal
        dist_x = abs(ex - CENTER)
        dist_y = abs(ey - CENTER)
        if dist_x > 0 and dist_y > 0:
            # Move diagonally
            return ex + dx, ey + dy
        elif dist_x > 0:
            return ex + dx, ey
        elif dist_y > 0:
            return ex, ey + dy
        return ex, ey

    # ------------------------------------------------------------------
    # Fire
    # ------------------------------------------------------------------

    def _fire(self) -> float:
        """Fire projectile in facing direction. Returns reward."""
        direction = DIRECTIONS[self._facing_idx]
        ddx, ddy = DIR_DELTAS[direction]
        self._trail_dir = direction
        trail: list[tuple[int, int]] = []

        x, y = CENTER + ddx, CENTER + ddy
        hit_reward = 0.0
        hit_enemy_pos: tuple[int, int] | None = None

        while 1 <= x <= GRID_SIZE - 2 and 1 <= y <= GRID_SIZE - 2:
            if (x, y) in self._enemies:
                hit_enemy_pos = (x, y)
                trail.append((x, y))
                break
            trail.append((x, y))
            x += ddx
            y += ddy

        if hit_enemy_pos is not None:
            self._enemies.remove(hit_enemy_pos)
            self._kills += 1
            hit_reward = 1.0
            self._message = "Hit!"

        self._trail = trail
        return hit_reward

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(GRID_SIZE, GRID_SIZE, SYM_FLOOR)

        # Draw border
        for x in range(GRID_SIZE):
            grid[0][x] = SYM_BORDER
            grid[GRID_SIZE - 1][x] = SYM_BORDER
        for y in range(GRID_SIZE):
            grid[y][0] = SYM_BORDER
            grid[y][GRID_SIZE - 1] = SYM_BORDER

        # Draw bullet trail
        trail_char = TRAIL_CHARS[self._trail_dir]
        for tx, ty in self._trail:
            grid[ty][tx] = trail_char

        # Draw enemies
        for ex, ey in self._enemies:
            grid[ey][ex] = SYM_ENEMY

        # Draw cannon
        grid[CENTER][CENTER] = SYM_CANNON

        direction = DIRECTIONS[self._facing_idx]
        legend_symbols: dict[str, str] = {
            SYM_CANNON: "your cannon",
            SYM_ENEMY: "enemy",
            SYM_FLOOR: "floor",
            SYM_BORDER: "border",
        }
        # Add trail chars to legend only when trail is visible
        if self._trail:
            legend_symbols[trail_char] = "bullet trail"

        legend = build_legend(legend_symbols)

        total_enemies_this_wave = self._wave_enemy_count(self._current_wave)
        enemies_left_to_spawn = total_enemies_this_wave - self._enemies_spawned_this_wave

        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Facing: {direction} {DIR_ARROWS[direction]}    "
            f"Wave: {self._current_wave + 1}/{TOTAL_WAVES}    "
            f"Enemies on field: {len(self._enemies)}    "
            f"Enemies incoming: {enemies_left_to_spawn}    "
            f"Kills: {self._kills}"
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
            "You control a rotating cannon on a 15x15 grid. Enemies approach "
            "from the grid edges toward your position. Destroy all enemies "
            "across 5 waves to win.\n\n"
            "RULES\n"
            "- The cannon faces one of 8 directions: "
            "N, NE, E, SE, S, SW, W, NW.\n"
            "- ROTATE_CW / ROTATE_CCW rotates 45 degrees.\n"
            "- FIRE shoots instantly in the facing direction, destroying the first "
            "enemy hit.\n"
            "- Enemies spawn at the grid edges every 3 steps and move 1 cell toward "
            "the center each step.\n"
            "- Wave N has N+2 enemies. There are 5 waves total.\n"
            "- If any enemy reaches the center, you lose (-1 reward).\n"
            "- If all 5 waves are cleared, you win (+1 reward).\n\n"
            "REWARDS\n"
            "- +1 per enemy killed\n"
            "- +5 per wave cleared\n"
            "- +1 for clearing all waves (victory)\n"
            "- penalty 1 if an enemy reaches the center (defeat)\n\n"
            + self.action_spec.render_for_prompt()
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

