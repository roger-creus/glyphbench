"""Atari Ice Hockey environment.

Top-down hockey rink. Score goals against AI opponent.

Gym ID: atlas_rl/atari-icehockey-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec

from .base import AtariBase


class IceHockeyEnv(AtariBase):
    """Ice Hockey: top-down hockey rink.

    20x24 grid. Score goals past the opponent goalie.
    AI opponent chases puck and shoots.

    Actions: NOOP, UP, DOWN, LEFT, RIGHT, SHOOT
    Reward: +1 per goal scored, -1 per goal conceded
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "UP", "DOWN", "LEFT",
            "RIGHT", "SHOOT",
        ),
        descriptions=(
            "do nothing",
            "move up (skate)",
            "move down (skate)",
            "move left (skate)",
            "move right (skate)",
            "shoot the puck",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 24
    _RINK_L = 1
    _RINK_R = 18
    _RINK_T = 1
    _RINK_B = 22
    _GOAL_L = 7
    _GOAL_R = 12
    _PERIOD_LEN = 500

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._opp_x: int = 0
        self._opp_y: int = 0
        self._puck_x: float = 0.0
        self._puck_y: float = 0.0
        self._puck_dx: float = 0.0
        self._puck_dy: float = 0.0
        self._puck_free: bool = True
        self._puck_holder: str = ""  # "player"/"opp"/""
        self._player_goals: int = 0
        self._opp_goals: int = 0
        self._period: int = 1
        self._period_timer: int = 0

    def env_id(self) -> str:
        return "atlas_rl/atari-icehockey-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._player_goals = 0
        self._opp_goals = 0
        self._period = 1
        self._period_timer = 0
        self._lives = 99
        self._reset_positions()
        self._redraw()

    def _reset_positions(self) -> None:
        cx = self._WIDTH // 2
        self._player_x, self._player_y = cx, self._RINK_B - 4
        self._opp_x, self._opp_y = cx, self._RINK_T + 4
        self._puck_x, self._puck_y = float(cx), float(self._HEIGHT // 2)
        self._puck_dx = self._puck_dy = 0.0
        self._puck_free, self._puck_holder = True, ""

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._period_timer += 1

        # Player movement
        nx, ny = self._player_x, self._player_y
        if action_name == "UP":
            ny -= 1
        elif action_name == "DOWN":
            ny += 1
        elif action_name == "LEFT":
            nx -= 1
        elif action_name == "RIGHT":
            nx += 1
        if self._in_rink(nx, ny):
            self._player_x, self._player_y = nx, ny

        # Pick up puck if nearby
        px = int(round(self._puck_x))
        py = int(round(self._puck_y))
        if (
            self._puck_free
            and abs(self._player_x - px) <= 1
            and abs(self._player_y - py) <= 1
        ):
            self._puck_holder = "player"
            self._puck_free = False

        # Carry puck
        if self._puck_holder == "player":
            self._puck_x = float(self._player_x)
            self._puck_y = float(self._player_y - 1)

        # Shoot
        if action_name == "SHOOT" and self._puck_holder == "player":
            self._puck_holder = ""
            self._puck_free = True
            self._puck_dy = -2.0
            self._puck_dx = float(self.rng.integers(-1, 2)) * 0.5

        # Move puck if free and moving
        if self._puck_free and (self._puck_dx or self._puck_dy):
            self._puck_x += self._puck_dx
            self._puck_y += self._puck_dy
            px = int(round(self._puck_x))
            py = int(round(self._puck_y))
            # Wall bounce
            if px <= self._RINK_L or px >= self._RINK_R:
                self._puck_dx = -self._puck_dx
            # Check goals
            if py <= self._RINK_T:
                if self._GOAL_L <= px <= self._GOAL_R:
                    self._player_goals += 1
                    self._on_point_scored(1)
                    reward = 1.0
                    self._message = "GOAL! You score!"
                    self._reset_positions()
                else:
                    self._puck_dy = -self._puck_dy
            elif py >= self._RINK_B:
                if self._GOAL_L <= px <= self._GOAL_R:
                    self._opp_goals += 1
                    reward = -1.0
                    self._message = "Opponent scores!"
                    self._reset_positions()
                else:
                    self._puck_dy = -self._puck_dy

            # Friction
            self._puck_dx *= 0.95
            self._puck_dy *= 0.95
            if abs(self._puck_dx) < 0.1 and abs(self._puck_dy) < 0.1:
                self._puck_dx = self._puck_dy = 0.0
        # Opponent AI
        self._move_opponent()
        # Period check
        terminated = False
        if self._period_timer >= self._PERIOD_LEN:
            self._period += 1
            self._period_timer = 0
            if self._period > 3:
                terminated = True
                if self._player_goals > self._opp_goals:
                    self._message = "You win!"
                elif self._opp_goals > self._player_goals:
                    self._message = "Opponent wins!"
                else:
                    self._message = "Draw!"
            else:
                self._reset_positions()

        info["player_goals"] = self._player_goals
        info["opp_goals"] = self._opp_goals
        info["period"] = min(self._period, 3)
        self._redraw()
        return reward, terminated, info

    def _in_rink(self, x: int, y: int) -> bool:
        return (
            self._RINK_L < x < self._RINK_R
            and self._RINK_T < y < self._RINK_B
        )

    def _move_opponent(self) -> None:
        rng = self.rng
        px = int(round(self._puck_x))
        py = int(round(self._puck_y))

        # Pick up puck
        if (
            self._puck_free
            and abs(self._opp_x - px) <= 1
            and abs(self._opp_y - py) <= 1
        ):
            self._puck_holder = "opp"
            self._puck_free = False

        if self._puck_holder == "opp":
            self._puck_x = float(self._opp_x)
            self._puck_y = float(self._opp_y + 1)
            # Shoot toward goal
            if (
                self._opp_y > self._RINK_B - 8
                and rng.random() < 0.3
            ):
                self._puck_holder = ""
                self._puck_free = True
                self._puck_dy = 2.0
                self._puck_dx = float(rng.integers(-1, 2)) * 0.5
        # Chase puck or drive to goal
        tx = px if self._puck_holder != "opp" else self._WIDTH // 2
        ty = py if self._puck_holder != "opp" else self._RINK_B
        if rng.random() < 0.6:
            dx = 0 if self._opp_x == tx else (1 if self._opp_x < tx else -1)
            dy = 0 if self._opp_y == ty else (1 if self._opp_y < ty else -1)
            nx, ny = self._opp_x + dx, self._opp_y + dy
            if self._in_rink(nx, ny):
                self._opp_x, self._opp_y = nx, ny

    def _redraw(self) -> None:
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                self._set_cell(x, y, " ")

        # Rink borders
        for x in range(self._RINK_L, self._RINK_R + 1):
            self._set_cell(x, self._RINK_T, "-")
            self._set_cell(x, self._RINK_B, "-")
        for y in range(self._RINK_T, self._RINK_B + 1):
            self._set_cell(self._RINK_L, y, "|")
            self._set_cell(self._RINK_R, y, "|")

        # Goals (gaps in border)
        for x in range(self._GOAL_L, self._GOAL_R + 1):
            self._set_cell(x, self._RINK_T, "G")
            self._set_cell(x, self._RINK_B, "G")

        # Center line
        for x in range(self._RINK_L + 1, self._RINK_R):
            self._set_cell(x, self._HEIGHT // 2, ".")

        # Puck
        px, py = int(round(self._puck_x)), int(round(self._puck_y))
        if self._in_rink(px, py):
            self._set_cell(px, py, "*")
        self._set_cell(self._opp_x, self._opp_y, "V")

    def _advance_entities(self) -> None:
        pass

    def _render_current_observation(self, **kw: Any):  # type: ignore[override]
        from atlas_rl.core.ascii_primitives import build_legend, grid_to_string
        from atlas_rl.core.observation import GridObservation
        render = [row[:] for row in self._grid]
        symbols: dict[str, str] = {}
        for y in range(self._grid_h):
            for x in range(self._grid_w):
                ch = render[y][x]
                if ch not in symbols:
                    symbols[ch] = self._symbol_meaning(ch)
        r, c = self._player_y, self._player_x
        if 0 <= c < self._grid_w and 0 <= r < self._grid_h:
            render[r][c] = "@"
        symbols["@"] = "you"
        p = min(self._period, 3)
        hud = f"You {self._player_goals} - {self._opp_goals} Opp | Period {p}/3"
        return GridObservation(
            grid=grid_to_string(render), legend=build_legend(symbols),
            hud=hud, message=self._message,
        )

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "-": "boards",
            "|": "boards",
            "G": "goal",
            ".": "center line",
            "*": "puck",
            "V": "opponent",
            " ": "ice",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Play ice hockey. Skate to pick up the puck, "
            "then SHOOT to score. "
            "Defend your goal from the opponent. "
            "3 periods, most goals wins."
        )
