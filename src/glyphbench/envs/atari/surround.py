"""Atari Surround environment.

Tron light-cycles game. Leave a trail, avoid crashing.

Gym ID: glyphbench/atari-surround-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase


class SurroundEnv(AtariBase):
    """Surround: Tron-like trail game.

    20x20 arena. Agent and AI opponent each leave trails.
    Hitting any trail (including your own) or a wall = death.

    Actions: NOOP, UP, RIGHT, LEFT, DOWN
    Reward: +1 for win, -1 for loss
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "RIGHT", "LEFT", "DOWN"),
        descriptions=(
            "continue in current direction",
            "turn up",
            "turn right",
            "turn left",
            "turn down",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._opp_x: int = 0
        self._opp_y: int = 0
        self._opp_dx: int = 0
        self._opp_dy: int = 0
        self._player_dx: int = 0
        self._player_dy: int = 0
        self._trail_player: set[tuple[int, int]] = set()
        self._trail_opp: set[tuple[int, int]] = set()

    def env_id(self) -> str:
        return "glyphbench/atari-surround-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._trail_player = set()
        self._trail_opp = set()

        # Border
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "█")
            self._set_cell(x, self._HEIGHT - 1, "█")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "█")
            self._set_cell(self._WIDTH - 1, y, "█")

        # Player starts left side, moving right
        self._player_x = self._WIDTH // 4
        self._player_y = self._HEIGHT // 2
        self._player_dx = 1
        self._player_dy = 0
        self._trail_player.add((self._player_x, self._player_y))

        # Opponent starts right side, moving left
        self._opp_x = 3 * self._WIDTH // 4
        self._opp_y = self._HEIGHT // 2
        self._opp_dx = -1
        self._opp_dy = 0
        self._trail_opp.add((self._opp_x, self._opp_y))

        self._redraw()

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Update player direction (no 180-degree turns)
        if action_name == "UP" and self._player_dy != 1:
            self._player_dx, self._player_dy = 0, -1
            self._player_dir = (0, -1)
        elif action_name == "DOWN" and self._player_dy != -1:
            self._player_dx, self._player_dy = 0, 1
            self._player_dir = (0, 1)
        elif action_name == "LEFT" and self._player_dx != 1:
            self._player_dx, self._player_dy = -1, 0
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._player_dx != -1:
            self._player_dx, self._player_dy = 1, 0
            self._player_dir = (1, 0)
        # NOOP: keep current direction

        # Move player
        new_px = self._player_x + self._player_dx
        new_py = self._player_y + self._player_dy

        # Move opponent (simple AI: try to move toward player, avoid trails)
        self._move_opponent()
        new_ox = self._opp_x + self._opp_dx
        new_oy = self._opp_y + self._opp_dy

        # Check collisions
        player_dead = self._is_collision(new_px, new_py)
        opp_dead = self._is_collision(new_ox, new_oy)

        if player_dead and opp_dead:
            reward = 0.0
            self._message = "Draw!"
            return reward, True, info
        elif player_dead:
            reward = -1.0
            self._message = "You crashed! -1"
            return reward, True, info
        elif opp_dead:
            reward = 1.0
            self._on_point_scored(1)
            self._message = "Opponent crashed! +1"
            return reward, True, info

        # Update positions and trails
        self._player_x = new_px
        self._player_y = new_py
        self._trail_player.add((new_px, new_py))

        self._opp_x = new_ox
        self._opp_y = new_oy
        self._trail_opp.add((new_ox, new_oy))

        self._redraw()
        return reward, False, info

    def _is_collision(self, x: int, y: int) -> bool:
        """Check if position hits a wall, trail, or boundary."""
        if x <= 0 or x >= self._WIDTH - 1 or y <= 0 or y >= self._HEIGHT - 1:
            return True
        return (x, y) in self._trail_player or (x, y) in self._trail_opp

    def _move_opponent(self) -> None:
        """Simple AI: try to chase player, avoid immediate death."""
        rng = self.rng

        # Possible directions
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        # Remove 180-degree turn
        directions = [
            (ddx, ddy) for ddx, ddy in directions
            if not (ddx == -self._opp_dx and ddy == -self._opp_dy)
        ]

        # Filter out deadly moves
        safe_dirs = []
        for ddx, ddy in directions:
            nx = self._opp_x + ddx
            ny = self._opp_y + ddy
            if not self._is_collision(nx, ny):
                safe_dirs.append((ddx, ddy))

        if safe_dirs:
            # Prefer direction toward player (with some randomness)
            if rng.random() < 0.6:
                # Sort by distance to player
                safe_dirs.sort(
                    key=lambda d: abs(self._opp_x + d[0] - self._player_x)
                    + abs(self._opp_y + d[1] - self._player_y)
                )
                self._opp_dx, self._opp_dy = safe_dirs[0]
            else:
                idx = int(rng.integers(len(safe_dirs)))
                self._opp_dx, self._opp_dy = safe_dirs[idx]
        # If no safe dirs, keep current direction (will crash)

    def _redraw(self) -> None:
        """Redraw trails and positions."""
        # Clear interior
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")

        # Draw player trail
        for tx, ty in self._trail_player:
            if 0 < tx < self._WIDTH - 1 and 0 < ty < self._HEIGHT - 1:
                self._set_cell(tx, ty, "┼")

        # Draw opponent trail
        for tx, ty in self._trail_opp:
            if 0 < tx < self._WIDTH - 1 and 0 < ty < self._HEIGHT - 1:
                self._set_cell(tx, ty, "x")

        # Draw opponent head
        if 0 < self._opp_x < self._WIDTH - 1 and 0 < self._opp_y < self._HEIGHT - 1:
            self._set_cell(self._opp_x, self._opp_y, "O")

    def _advance_entities(self) -> None:
        # No entities to advance
        pass

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "█": "wall",
            "┼": "your trail",
            "x": "opponent trail",
            "O": "opponent",
            " ": "empty",
        }.get(ch, ch)

    _DIR_MAP = {
        (0, -1): "up", (0, 1): "down",
        (-1, 0): "left", (1, 0): "right",
    }

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        p_dir = self._DIR_MAP.get(
            (self._player_dx, self._player_dy), "?"
        )
        o_dir = self._DIR_MAP.get(
            (self._opp_dx, self._opp_dy), "?"
        )
        extra = (
            f"Your dir: {p_dir}"
            f"  Opponent dir: {o_dir}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Leave a trail behind you and try to make the opponent crash. "
            "Avoid hitting any wall, your own trail, or the opponent's trail. "
            "You cannot reverse direction."
        )
