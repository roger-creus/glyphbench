"""Atari Q*bert environment.

Hop on cubes in a pyramid to change their color. Avoid enemies.

Gym ID: glyphbench/atari-qbert-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase


class QbertEnv(AtariBase):
    """Q*bert: hop on cubes in a pyramid to change their color.

    The pyramid has 7 rows (1 cube on top, 7 on bottom = 28 cubes).
    Mapped to a 15x15 grid. Enemies descend periodically.

    Actions: NOOP, UP_RIGHT, UP_LEFT, DOWN_RIGHT, DOWN_LEFT
    Reward: +25 per cube colored, +100 level clear
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP_RIGHT", "UP_LEFT", "DOWN_RIGHT", "DOWN_LEFT"),
        descriptions=(
            "do nothing",
            "hop diagonally up-right",
            "hop diagonally up-left",
            "hop diagonally down-right",
            "hop diagonally down-left",
        ),
    )

    _WIDTH = 15
    _HEIGHT = 15
    _NUM_ROWS = 7
    _ENEMY_SPAWN_INTERVAL = 8

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        # (row, col) -> colored?  row 0 is top of pyramid
        self._cubes: dict[tuple[int, int], bool] = {}
        # Mapping from (row, col) to grid (x, y)
        self._cube_positions: dict[tuple[int, int], tuple[int, int]] = {}
        self._player_row: int = 0
        self._player_col: int = 0
        self._step_counter: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-qbert-v0"

    def _build_cube_positions(self) -> None:
        """Compute grid coordinates for each cube in the pyramid."""
        self._cube_positions = {}
        for row in range(self._NUM_ROWS):
            num_cubes = row + 1
            # Center the row horizontally
            start_x = self._WIDTH // 2 - row
            y = 2 + row * 2
            for col in range(num_cubes):
                gx = start_x + col * 2
                self._cube_positions[(row, col)] = (gx, y)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._step_counter = 0

        self._build_cube_positions()
        self._cubes = {pos: False for pos in self._cube_positions}

        # Draw pyramid cubes on grid
        self._redraw_pyramid()

        # Player starts at the top of the pyramid
        self._player_row = 0
        self._player_col = 0
        gx, gy = self._cube_positions[(0, 0)]
        self._player_x = gx
        self._player_y = gy

    def _redraw_pyramid(self) -> None:
        """Redraw all cubes on the grid."""
        # Clear interior
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                self._set_cell(x, y, " ")

        for (row, col), colored in self._cubes.items():
            gx, gy = self._cube_positions[(row, col)]
            ch = "█" if colored else "·"
            self._set_cell(gx, gy, ch)

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        new_row = self._player_row
        new_col = self._player_col

        if action_name == "UP_RIGHT":
            new_row -= 1
            self._player_dir = (1, 0)
            # col stays same (up-right on pyramid)
        elif action_name == "UP_LEFT":
            new_row -= 1
            new_col -= 1
            self._player_dir = (-1, 0)
        elif action_name == "DOWN_RIGHT":
            new_row += 1
            new_col += 1
            self._player_dir = (1, 0)
        elif action_name == "DOWN_LEFT":
            new_row += 1
            self._player_dir = (-1, 0)
            # col stays same (down-left on pyramid)

        if action_name != "NOOP":
            if (new_row, new_col) in self._cube_positions:
                self._player_row = new_row
                self._player_col = new_col
                gx, gy = self._cube_positions[(new_row, new_col)]
                self._player_x = gx
                self._player_y = gy

                # Color the cube
                if not self._cubes[(new_row, new_col)]:
                    self._cubes[(new_row, new_col)] = True
                    self._on_point_scored(25)
                    reward += 25
                    self._message = "Cube colored! +25"
            else:
                # Fell off the pyramid
                self._on_life_lost()
                self._message = "Fell off! Lost a life."
                # Reset to top
                self._player_row = 0
                self._player_col = 0
                gx, gy = self._cube_positions[(0, 0)]
                self._player_x = gx
                self._player_y = gy

        # Check level complete
        if all(self._cubes.values()):
            self._on_point_scored(100)
            reward += 100
            self._message = "Level clear! +100"
            self._level += 1
            self._cubes = {pos: False for pos in self._cube_positions}

        # Spawn enemies periodically
        if self._step_counter % self._ENEMY_SPAWN_INTERVAL == 0:
            self._spawn_enemy()

        # Move and check enemy collisions
        self._move_enemies()

        self._redraw_pyramid()

        info["cubes_colored"] = sum(1 for v in self._cubes.values() if v)
        info["cubes_total"] = len(self._cubes)
        return reward, self._game_over, info

    def _spawn_enemy(self) -> None:
        """Spawn an enemy at the top of the pyramid."""
        top_x, top_y = self._cube_positions[(0, 0)]
        etype = "snake" if self.rng.random() < 0.5 else "ball"
        char = "s" if etype == "snake" else "o"
        e = self._add_entity(etype, char, top_x, top_y)
        e.data["prow"] = 0
        e.data["pcol"] = 0

    def _move_enemies(self) -> None:
        """Move enemies down the pyramid."""
        for e in self._entities:
            if not e.alive:
                continue
            prow = e.data.get("prow", 0)
            pcol = e.data.get("pcol", 0)

            # Move down randomly
            if self.rng.random() < 0.5:
                new_row = prow + 1
                new_col = pcol + 1
            else:
                new_row = prow + 1
                new_col = pcol

            if (new_row, new_col) in self._cube_positions:
                e.data["prow"] = new_row
                e.data["pcol"] = new_col
                gx, gy = self._cube_positions[(new_row, new_col)]
                e.x = gx
                e.y = gy

                # Check collision with player
                if new_row == self._player_row and new_col == self._player_col:
                    self._on_life_lost()
                    self._message = "Hit by enemy! Lost a life."
                    self._player_row = 0
                    self._player_col = 0
                    gx2, gy2 = self._cube_positions[(0, 0)]
                    self._player_x = gx2
                    self._player_y = gy2
                    e.alive = False
            else:
                e.alive = False

    def _advance_entities(self) -> None:
        # Enemies are moved in _game_step
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "·": "uncolored cube",
            "█": "colored cube",
            "s": "snake (enemy)",
            "o": "ball (enemy)",
            " ": "empty",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        colored = sum(
            1 for v in self._cubes.values() if v
        )
        total = len(self._cubes)
        extra = f"Cubes: {colored}/{total}"
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Hop on all cubes in the pyramid to change their color. "
            "Avoid enemies (snakes and balls). "
            "Don't jump off the edge of the pyramid."
        )
