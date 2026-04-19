"""Atari Video Pinball environment.

Pinball machine with flippers, bumpers, and targets.

Gym ID: atlas_rl/atari-videopinball-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec

from .base import AtariBase, AtariEntity


class VideoPinballEnv(AtariBase):
    """Video Pinball: pinball with flippers and bumpers.

    12x24 grid. Ball bounces around hitting bumpers and targets.
    Use flippers to keep ball in play. Nudge to influence ball.

    Actions: NOOP, LEFT_FLIPPER, RIGHT_FLIPPER, NUDGE
    Reward: +10 bumper, +50 target, +100 spinner
    Lives: 3 (lost when ball drains)
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT_FLIPPER", "RIGHT_FLIPPER", "NUDGE"),
        descriptions=(
            "do nothing",
            "activate left flipper",
            "activate right flipper",
            "nudge the table (shifts ball left)",
        ),
    )

    _WIDTH = 12
    _HEIGHT = 24
    _FLIPPER_Y = 21
    _DRAIN_Y = 22

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._ball_x: int = 0
        self._ball_y: int = 0
        self._ball_dx: int = 0
        self._ball_dy: int = 0
        self._bumpers: list[tuple[int, int]] = []
        self._targets: list[AtariEntity] = []
        self._spinners: list[AtariEntity] = []
        self._step_counter: int = 0
        self._ball_launched: bool = False

    def env_id(self) -> str:
        return "atlas_rl/atari-videopinball-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._targets = []
        self._spinners = []
        self._bumpers = []
        self._step_counter = 0
        self._ball_launched = True

        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "-")
            self._set_cell(x, self._HEIGHT - 1, "-")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "|")
            self._set_cell(self._WIDTH - 1, y, "|")

        # Flippers
        self._set_cell(3, self._FLIPPER_Y, "\\")
        self._set_cell(4, self._FLIPPER_Y, "_")
        self._set_cell(self._WIDTH - 5, self._FLIPPER_Y, "_")
        self._set_cell(self._WIDTH - 4, self._FLIPPER_Y, "/")

        # Drain
        for x in range(1, self._WIDTH - 1):
            self._set_cell(x, self._DRAIN_Y, ".")

        rng = self.rng

        # Bumpers (round objects that bounce ball)
        bumper_spots = [
            (4, 6), (7, 6),
            (3, 10), (8, 10), (5, 8),
            (6, 12),
        ]
        for bx, by in bumper_spots:
            if bx < self._WIDTH - 1 and by < self._HEIGHT - 1:
                self._bumpers.append((bx, by))

        # Targets (hit to destroy for points)
        for _i in range(3 + self._level):
            for _ in range(20):
                tx = int(rng.integers(2, self._WIDTH - 2))
                ty = int(rng.integers(3, 15))
                if (tx, ty) not in self._bumpers:
                    t = self._add_entity(
                        "target", "X", tx, ty
                    )
                    self._targets.append(t)
                    break

        # Spinners
        for sx in (3, self._WIDTH - 4):
            if 0 < sx < self._WIDTH - 1:
                sp = self._add_entity("spinner", "S", sx, 4)
                self._spinners.append(sp)

        # Ball start position
        self._ball_x = self._WIDTH // 2
        self._ball_y = self._FLIPPER_Y - 1
        self._ball_dx = int(rng.choice([-1, 1]))
        self._ball_dy = -1

        # Player position (not rendered separately)
        self._player_x = self._WIDTH // 2
        self._player_y = self._FLIPPER_Y

        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        # Flipper actions
        left_flip = action_name == "LEFT_FLIPPER"
        right_flip = action_name == "RIGHT_FLIPPER"
        nudge = action_name == "NUDGE"

        # Move ball
        nx = self._ball_x + self._ball_dx
        ny = self._ball_y + self._ball_dy

        # Wall bounces
        if nx <= 0 or nx >= self._WIDTH - 1:
            self._ball_dx = -self._ball_dx
            nx = self._ball_x + self._ball_dx
        if ny <= 0:
            self._ball_dy = -self._ball_dy
            ny = self._ball_y + self._ball_dy

        # Flipper collision
        if ny >= self._FLIPPER_Y:
            if left_flip and 2 <= nx <= 5:
                self._ball_dy = -1
                self._ball_dx = 1 if nx >= 4 else -1
                ny = self._FLIPPER_Y - 1
                self._message = "Left flipper!"
            elif right_flip and (
                self._WIDTH - 6 <= nx <= self._WIDTH - 3
            ):
                self._ball_dy = -1
                self._ball_dx = -1 if nx <= self._WIDTH - 5 else 1
                ny = self._FLIPPER_Y - 1
                self._message = "Right flipper!"
            elif ny >= self._DRAIN_Y:
                # Ball drained
                self._on_life_lost()
                self._message = "Ball drained!"
                self._ball_x = self._WIDTH // 2
                self._ball_y = self._FLIPPER_Y - 1
                rng = self.rng
                self._ball_dx = int(rng.choice([-1, 1]))
                self._ball_dy = -1
                self._redraw()
                return reward, self._game_over, info

        # Nudge shifts ball
        if nudge:
            nx = max(1, nx - 1)

        self._ball_x = max(1, min(nx, self._WIDTH - 2))
        self._ball_y = max(1, min(ny, self._HEIGHT - 2))

        # Bumper collision
        for bx, by in self._bumpers:
            if (
                self._ball_x == bx
                and self._ball_y == by
            ):
                self._ball_dx = -self._ball_dx
                self._ball_dy = -self._ball_dy
                self._on_point_scored(10)
                reward += 10
                self._message = "Bumper! +10"
                # Bounce away
                self._ball_x += self._ball_dx
                self._ball_y += self._ball_dy
                break

        # Target collision
        for t in self._targets:
            if (
                t.alive
                and t.x == self._ball_x
                and t.y == self._ball_y
            ):
                t.alive = False
                self._ball_dy = -self._ball_dy
                self._on_point_scored(50)
                reward += 50
                self._message = "Target! +50"

        # Spinner collision
        for sp in self._spinners:
            if (
                sp.alive
                and sp.x == self._ball_x
                and sp.y == self._ball_y
            ):
                self._ball_dy = -self._ball_dy
                self._on_point_scored(100)
                reward += 100
                self._message = "Spinner! +100"

        # All targets cleared -> respawn
        alive_targets = [
            t for t in self._targets if t.alive
        ]
        if len(alive_targets) == 0 and self._targets:
            self._level += 1
            self._message = "All targets cleared!"
            # Respawn targets
            rng = self.rng
            self._targets = []
            for _i in range(3 + self._level):
                for _ in range(20):
                    tx = int(rng.integers(2, self._WIDTH - 2))
                    ty = int(rng.integers(3, 15))
                    if (tx, ty) not in self._bumpers:
                        t = self._add_entity(
                            "target", "X", tx, ty
                        )
                        self._targets.append(t)
                        break

        # Cleanup
        self._targets = [
            t for t in self._targets if t.alive
        ]

        self._redraw()
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")
        # Flippers
        self._set_cell(3, self._FLIPPER_Y, "\\")
        self._set_cell(4, self._FLIPPER_Y, "_")
        self._set_cell(
            self._WIDTH - 5, self._FLIPPER_Y, "_"
        )
        self._set_cell(
            self._WIDTH - 4, self._FLIPPER_Y, "/"
        )
        # Drain
        for x in range(1, self._WIDTH - 1):
            self._set_cell(x, self._DRAIN_Y, ".")
        # Bumpers
        for bx, by in self._bumpers:
            self._set_cell(bx, by, "O")
        # Targets
        for t in self._targets:
            if t.alive:
                self._set_cell(t.x, t.y, "X")
        # Spinners
        for sp in self._spinners:
            if sp.alive:
                self._set_cell(sp.x, sp.y, "S")
        # Ball
        if (
            1 <= self._ball_x < self._WIDTH - 1
            and 1 <= self._ball_y < self._HEIGHT - 1
        ):
            self._set_cell(self._ball_x, self._ball_y, "o")

    def _render_current_observation(self, **kw):  # type: ignore[override]
        # Hide player marker (ball is the focus)
        old_px, old_py = self._player_x, self._player_y
        self._player_x = -1
        self._player_y = -1
        obs = super()._render_current_observation()
        self._player_x = old_px
        self._player_y = old_py
        return obs

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "-": "wall",
            "|": "wall",
            "\\": "left flipper",
            "/": "right flipper",
            "_": "flipper",
            ".": "drain (ball lost here)",
            "O": "bumper (10pts)",
            "X": "target (50pts)",
            "S": "spinner (100pts)",
            "o": "ball",
            " ": "empty",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Keep the ball (o) in play using flippers. "
            "Hit bumpers (O) for 10pts, targets (X) for 50pts, "
            "spinners (S) for 100pts. "
            "Ball draining past flippers costs a life. "
            "NUDGE shifts the ball left."
        )
