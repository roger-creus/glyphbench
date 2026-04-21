"""Atari Boxing environment.

Top-down boxing ring. Two boxers trade punches.

Gym ID: glyphbench/atari-boxing-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec

from .base import AtariBase


class BoxingEnv(AtariBase):
    """Boxing: two boxers in a ring.

    20x20 grid. Land punches on the opponent.
    First to 100 or most punches when time runs out.

    Actions: NOOP, UP, DOWN, LEFT, RIGHT, PUNCH
    Reward: +1 per punch landed
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "UP", "DOWN", "LEFT", "RIGHT", "PUNCH",
        ),
        descriptions=(
            "do nothing",
            "move up",
            "move down",
            "move left",
            "move right",
            "throw a punch",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _RING_L = 2
    _RING_R = 17
    _RING_T = 2
    _RING_B = 17
    _WIN_SCORE = 100

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._opp_x: int = 0
        self._opp_y: int = 0
        self._opp_score: int = 0
        self._punch_cooldown: int = 0
        self._opp_cooldown: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-boxing-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._opp_score = 0
        self._punch_cooldown = 0
        self._opp_cooldown = 0

        # Player bottom-center, opponent top-center
        self._player_x = self._WIDTH // 2
        self._player_y = self._RING_B - 2
        self._opp_x = self._WIDTH // 2
        self._opp_y = self._RING_T + 2
        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        if self._punch_cooldown > 0:
            self._punch_cooldown -= 1
        if self._opp_cooldown > 0:
            self._opp_cooldown -= 1

        # Player movement
        nx, ny = self._player_x, self._player_y
        if action_name == "UP":
            ny -= 1
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            ny += 1
            self._player_dir = (0, 1)
        elif action_name == "LEFT":
            nx -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            nx += 1
            self._player_dir = (1, 0)

        if self._in_ring(nx, ny) and (
            nx != self._opp_x or ny != self._opp_y
        ):
            self._player_x, self._player_y = nx, ny

        # Player punch
        if (
            action_name == "PUNCH"
            and self._punch_cooldown == 0
        ):
            self._punch_cooldown = 3
            dist = (
                abs(self._player_x - self._opp_x)
                + abs(self._player_y - self._opp_y)
            )
            if dist <= 2:
                self._on_point_scored(1)
                reward = 1.0
                self._message = "Punch landed! +1"

        # Opponent AI
        self._move_opponent()

        # Opponent punch
        if self._opp_cooldown == 0:
            dist = (
                abs(self._player_x - self._opp_x)
                + abs(self._player_y - self._opp_y)
            )
            if dist <= 2 and self.rng.random() < 0.4:
                self._opp_cooldown = 3
                self._opp_score += 1
                self._message = "Opponent punched you!"

        # Check win
        terminated = False
        if self._score >= self._WIN_SCORE:
            self._message = "You win by KO!"
            self._game_over = True
            terminated = True
        elif self._opp_score >= self._WIN_SCORE:
            self._message = "Opponent wins by KO!"
            self._game_over = True
            terminated = True
            reward = -1.0

        info["opp_score"] = self._opp_score
        self._redraw()
        return reward, terminated, info

    def _in_ring(self, x: int, y: int) -> bool:
        return (
            self._RING_L < x < self._RING_R
            and self._RING_T < y < self._RING_B
        )

    def _move_opponent(self) -> None:
        rng = self.rng
        dx, dy = 0, 0
        if rng.random() < 0.5:
            # Chase player
            if self._opp_x < self._player_x:
                dx = 1
            elif self._opp_x > self._player_x:
                dx = -1
            if self._opp_y < self._player_y:
                dy = 1
            elif self._opp_y > self._player_y:
                dy = -1
        else:
            # Random movement
            dx = int(rng.integers(-1, 2))
            dy = int(rng.integers(-1, 2))

        nx = self._opp_x + dx
        ny = self._opp_y + dy
        if self._in_ring(nx, ny) and (
            nx != self._player_x or ny != self._player_y
        ):
            self._opp_x, self._opp_y = nx, ny

    def _redraw(self) -> None:
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                self._set_cell(x, y, " ")

        # Ring ropes
        for x in range(self._RING_L, self._RING_R + 1):
            self._set_cell(x, self._RING_T, "=")
            self._set_cell(x, self._RING_B, "=")
        for y in range(self._RING_T, self._RING_B + 1):
            self._set_cell(self._RING_L, y, "│")
            self._set_cell(self._RING_R, y, "│")
        self._set_cell(self._RING_L, self._RING_T, "┼")
        self._set_cell(self._RING_R, self._RING_T, "┼")
        self._set_cell(self._RING_L, self._RING_B, "┼")
        self._set_cell(self._RING_R, self._RING_B, "┼")

        # Opponent
        self._set_cell(self._opp_x, self._opp_y, "B")

    def _advance_entities(self) -> None:
        pass

    def _render_current_observation(self, **kw: Any):  # type: ignore[override]
        """Override to show opponent score in HUD."""
        from glyphbench.core.ascii_primitives import (
            build_legend,
            grid_to_string,
        )
        from glyphbench.core.observation import GridObservation

        render = [row[:] for row in self._grid]
        symbols: dict[str, str] = {}
        for y in range(self._grid_h):
            for x in range(self._grid_w):
                ch = render[y][x]
                if ch not in symbols:
                    symbols[ch] = self._symbol_meaning(ch)
        r = self._player_y
        c = self._player_x
        if 0 <= c < self._grid_w and 0 <= r < self._grid_h:
            pch = self._DIR_CHARS.get(
                self._player_dir, "@"
            )
            render[r][c] = pch
            dname = self._DIR_NAMES.get(
                self._player_dir, "none"
            )
            symbols[pch] = f"you (facing {dname})"
        hud = (
            f"Your punches: {self._score}  "
            f"Opp punches: {self._opp_score}  "
            f"First to {self._WIN_SCORE}"
        )
        return GridObservation(
            grid=grid_to_string(render),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "=": "ring rope",
            "│": "ring rope",
            "┼": "corner post",
            "B": "opponent boxer",
            " ": "ring floor",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Box your opponent in the ring. "
            "Move close then PUNCH to land a hit. "
            f"First to {self._WIN_SCORE} punches wins."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Boxing.\n\n"
            "TASK\n"
            "Face off against an AI boxer in a square ring. Whoever "
            "lands 100 punches first wins the match.\n\n"
            "BOARD\n"
            "20x20 grid. Ring ropes: horizontal '=' at rows 2 and 17, "
            "vertical '|' at columns 2 and 17, corner posts '+'. You "
            "start near bottom-center (row 15) as an arrow glyph; the "
            "opponent 'B' starts near top-center (row 4). Both pieces "
            "stay within the ring interior (x in 3..16, y in 3..16).\n\n"
            "MECHANICS\n"
            "Each step you pick one of: NOOP, UP/DOWN/LEFT/RIGHT, or "
            "PUNCH. A move shifts you 1 cell (blocked by opponent "
            "position and ring walls). PUNCH lands if the Manhattan "
            "distance between you and the opponent is <= 2; PUNCH has "
            "a 3-step cooldown. Opponent moves each step (50 percent "
            "chase, 50 percent random) and punches when adjacent with "
            "40 percent chance, also on a 3-step cooldown.\n\n"
            "SCORING\n"
            "+1 reward per punch you land. No reward shaping for moving. "
            "Opponent punches increment their score but give reward 0 "
            "per-hit (only the KO gives -1 reward at game end). Winning "
            "by reaching 100 first terminates with no extra reward; "
            "losing terminates with -1 reward.\n\n"
            "TERMINATION\n"
            "First boxer to 100 punches wins (match ends). No life "
            "counter. Episode also ends after max_turns.\n\n"
            "HUD\n"
            "Shows your punch count, opponent punch count, and target "
            "score (100).\n\n"
            + self.action_spec.render_for_prompt()
        )
