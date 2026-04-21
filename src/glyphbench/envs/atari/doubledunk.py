"""Atari Double Dunk (basketball) environment.

Basketball game with 2-point and 3-point shots.

Gym ID: glyphbench/atari-doubledunk-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec

from .base import AtariBase


class DoubleDunkEnv(AtariBase):
    """Double Dunk: basketball game.

    20x16 court. Score baskets (2 or 3 points).
    AI opponent defends and attacks.

    Actions: NOOP, UP, DOWN, LEFT, RIGHT, SHOOT
    Reward: +2 or +3 per basket scored
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "UP", "DOWN", "LEFT",
            "RIGHT", "SHOOT",
        ),
        descriptions=(
            "do nothing",
            "move up",
            "move down",
            "move left",
            "move right",
            "shoot the ball",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 16
    _COURT_L = 1
    _COURT_R = 18
    _COURT_T = 1
    _COURT_B = 14
    _HOOP_Y = 2
    _OPP_HOOP_Y = 13
    _THREE_LINE = 5

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._opp_x: int = 0
        self._opp_y: int = 0
        self._opp_score: int = 0
        self._has_ball: bool = True
        self._ball_x: float = 0.0
        self._ball_y: float = 0.0
        self._ball_flying: bool = False
        self._ball_dx: float = 0.0
        self._ball_dy: float = 0.0
        self._shot_clock: int = 0
        self._quarter: int = 1
        self._quarter_timer: int = 0
        self._quarter_len: int = 300

    def env_id(self) -> str:
        return "glyphbench/atari-doubledunk-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._opp_score = 0
        self._quarter = 1
        self._quarter_timer = 0
        self._lives = 99
        self._reset_possession(player_has=True)
        self._redraw()

    def _reset_possession(self, player_has: bool = True) -> None:
        cx = self._WIDTH // 2
        self._player_x, self._player_y = cx, self._COURT_B - 3
        self._opp_x, self._opp_y = cx, self._COURT_T + 3
        self._has_ball, self._ball_flying = player_has, False
        self._shot_clock = 0

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._quarter_timer += 1

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
        if self._on_court(nx, ny):
            self._player_x, self._player_y = nx, ny

        # Ball follows player if held
        if self._has_ball and not self._ball_flying:
            self._ball_x = float(self._player_x)
            self._ball_y = float(self._player_y - 1)

        # Shoot
        hx = self._WIDTH // 2
        if action_name == "SHOOT" and self._has_ball and not self._ball_flying:
            self._ball_flying, self._has_ball = True, False
            self._ball_dx = (hx - self._player_x) * 0.3
            self._ball_dy = -1.5
        # Move flying ball
        if self._ball_flying:
            self._ball_x += self._ball_dx
            self._ball_y += self._ball_dy
            bx, by = int(round(self._ball_x)), int(round(self._ball_y))
            if by <= self._HOOP_Y + 1:
                if abs(bx - hx) <= 2:
                    dist = abs(self._player_y - self._HOOP_Y)
                    pts = 3 if dist >= self._THREE_LINE else 2
                    self._message = f"{'Three pointer' if pts == 3 else 'Basket'}! +{pts}"
                    if self.rng.random() < 0.6:
                        self._on_point_scored(pts)
                        reward = float(pts)
                    else:
                        self._message = "Missed shot!"
                    self._ball_flying = False
                    self._reset_possession(player_has=False)
                elif by < self._COURT_T:
                    self._ball_flying = False
                    self._message = "Air ball!"
                    self._reset_possession(player_has=False)

        # Opponent AI
        self._move_opponent()

        # Steal check
        if self._has_ball:
            dist = abs(self._player_x - self._opp_x) + abs(self._player_y - self._opp_y)
            if dist <= 1 and self.rng.random() < 0.15:
                self._has_ball = False
                self._message = "Stolen!"
                self._opp_attack()

        # Quarter transitions
        terminated = False
        if self._quarter_timer >= self._quarter_len:
            self._quarter += 1
            self._quarter_timer = 0
            if self._quarter > 4:
                terminated = True
                if self._score > self._opp_score:
                    self._message = "You win!"
                elif self._opp_score > self._score:
                    self._message = "Opponent wins!"
                else:
                    self._message = "Tie game!"
            else:
                self._reset_possession(
                    player_has=self._quarter % 2 == 1
                )
                self._message = (
                    f"Quarter {self._quarter}"
                )

        info["opp_score"] = self._opp_score
        info["quarter"] = min(self._quarter, 4)
        self._redraw()
        return reward, terminated, info

    def _on_court(self, x: int, y: int) -> bool:
        return (
            self._COURT_L < x < self._COURT_R
            and self._COURT_T < y < self._COURT_B
        )

    def _move_opponent(self) -> None:
        rng = self.rng
        def _dir(a: int, b: int) -> int:
            return 0 if a == b else (1 if a < b else -1)
        if self._has_ball or self._ball_flying:
            dx = _dir(self._opp_x, self._player_x)
            dy = _dir(self._opp_y, self._player_y)
        else:
            dx = _dir(self._opp_x, self._WIDTH // 2)
            dy = 1
        if rng.random() < 0.5:
            nx, ny = self._opp_x + dx, self._opp_y + dy
            if self._on_court(nx, ny):
                self._opp_x, self._opp_y = nx, ny

    def _opp_attack(self) -> None:
        rng = self.rng
        dist = abs(self._opp_y - self._OPP_HOOP_Y)
        if dist < 6 and rng.random() < 0.2:
            pts = 3 if dist >= self._THREE_LINE else 2
            if rng.random() < 0.4:
                self._opp_score += pts
                self._message = f"Opponent scores {pts}!"
            else:
                self._message = "Opponent missed!"
            self._reset_possession(player_has=True)

    def _redraw(self) -> None:
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                self._set_cell(x, y, " ")

        # Court borders
        for x in range(self._COURT_L, self._COURT_R + 1):
            self._set_cell(x, self._COURT_T, "─")
            self._set_cell(x, self._COURT_B, "─")
        for y in range(self._COURT_T, self._COURT_B + 1):
            self._set_cell(self._COURT_L, y, "│")
            self._set_cell(self._COURT_R, y, "│")

        # Hoops and half court
        hx, mid = self._WIDTH // 2, self._HEIGHT // 2
        self._set_cell(hx, self._HOOP_Y, "H")
        self._set_cell(hx, self._OPP_HOOP_Y, "H")
        for x in range(self._COURT_L + 1, self._COURT_R):
            self._set_cell(x, mid, "·")
        # Three-point arcs
        for x in range(hx - 4, hx + 5):
            if self._COURT_L < x < self._COURT_R:
                self._set_cell(x, self._HOOP_Y + self._THREE_LINE, "~")
                self._set_cell(x, self._OPP_HOOP_Y - self._THREE_LINE, "~")
        # Ball and opponent
        if self._ball_flying:
            bx, by = int(round(self._ball_x)), int(round(self._ball_y))
            if self._on_court(bx, by):
                self._set_cell(bx, by, "o")
        self._set_cell(self._opp_x, self._opp_y, "V")

    def _advance_entities(self) -> None:
        pass

    def _render_current_observation(self, **kw: Any):  # type: ignore[override]
        from glyphbench.core.ascii_primitives import build_legend, grid_to_string
        from glyphbench.core.observation import GridObservation
        render = [row[:] for row in self._grid]
        symbols: dict[str, str] = {}
        for y in range(self._grid_h):
            for x in range(self._grid_w):
                ch = render[y][x]
                if ch not in symbols:
                    symbols[ch] = self._symbol_meaning(ch)
        r, c = self._player_y, self._player_x
        if 0 <= c < self._grid_w and 0 <= r < self._grid_h:
            pch = self._DIR_CHARS.get(
                self._player_dir, "@"
            )
            render[r][c] = pch
            dname = self._DIR_NAMES.get(
                self._player_dir, "none"
            )
            symbols[pch] = f"you (facing {dname})"
        q = min(self._quarter, 4)
        if self._has_ball:
            ball_state = "held by you"
        elif self._ball_flying:
            bx = int(round(self._ball_x))
            by = int(round(self._ball_y))
            bdx = round(self._ball_dx, 1)
            bdy = round(self._ball_dy, 1)
            ball_state = (
                f"in flight pos=({bx},{by})"
                f" vel=({bdx},{bdy})"
            )
        else:
            ball_state = "held by opponent"
        hud = (
            f"You {self._score} - {self._opp_score} Opp"
            f" | Q{q}\nBall: {ball_state}"
        )
        return GridObservation(
            grid=grid_to_string(render), legend=build_legend(symbols),
            hud=hud, message=self._message,
        )

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "court line",
            "│": "court line",
            "H": "hoop",
            "·": "half court",
            "~": "three-point line",
            "o": "ball",
            "V": "opponent",
            " ": "court",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Play basketball. Move and SHOOT to score. "
            "Shots beyond the ~ line are worth 3 points, "
            "closer shots worth 2. "
            "4 quarters. Most points wins."
        )
