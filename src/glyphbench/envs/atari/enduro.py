"""Atari Enduro environment.

Racing game. Pass cars in lanes to score.

Gym ID: glyphbench/atari-enduro-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec

from .base import AtariBase, AtariEntity


class EnduroEnv(AtariBase):
    """Enduro: lane-based racing game.

    12x20 grid. Pass other cars to score. Avoid collisions.
    Must pass a target number of cars each day.

    Actions: NOOP, LEFT, RIGHT, ACCELERATE, BRAKE
    Reward: +1 per car passed
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "LEFT", "RIGHT",
            "ACCELERATE", "BRAKE",
        ),
        descriptions=(
            "do nothing",
            "move left one lane",
            "move right one lane",
            "speed up",
            "slow down",
        ),
    )

    _WIDTH = 12
    _HEIGHT = 20
    _NUM_LANES = 3
    _LANE_WIDTH = 3
    _ROAD_LEFT = 1
    _PLAYER_ROW = 17
    _CARS_TARGET = 20

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._player_lane: int = 1
        self._speed: int = 2
        self._cars_passed: int = 0
        self._day: int = 1
        self._spawn_timer: int = 0
        self._traffic: list[AtariEntity] = []

    def env_id(self) -> str:
        return "glyphbench/atari-enduro-v0"

    def _lane_x(self, lane: int) -> int:
        return (
            self._ROAD_LEFT
            + 1
            + lane * self._LANE_WIDTH
            + self._LANE_WIDTH // 2
        )

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._traffic = []
        self._player_lane = 1
        self._speed = 2
        self._cars_passed = 0
        self._day = 1
        self._spawn_timer = 0
        self._lives = 3

        self._player_x = self._lane_x(self._player_lane)
        self._player_y = self._PLAYER_ROW
        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Player controls
        if action_name == "LEFT":
            if self._player_lane > 0:
                self._player_lane -= 1
                self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            if self._player_lane < self._NUM_LANES - 1:
                self._player_lane += 1
                self._player_dir = (1, 0)
        elif action_name == "ACCELERATE":
            self._speed = min(4, self._speed + 1)
        elif action_name == "BRAKE":
            self._speed = max(1, self._speed - 1)

        self._player_x = self._lane_x(self._player_lane)

        # Spawn traffic
        self._spawn_timer += 1
        if self._spawn_timer >= max(3, 7 - self._speed):
            self._spawn_timer = 0
            rng = self.rng
            lane = int(rng.integers(0, self._NUM_LANES))
            cx = self._lane_x(lane)
            car = self._add_entity(
                "traffic", "C", cx, 1
            )
            car.data["lane"] = lane
            car.data["speed"] = 1
            self._traffic.append(car)

        # Move traffic (relative to player speed)
        drift = self._speed - 1
        to_remove: list[AtariEntity] = []
        for car in self._traffic:
            if not car.alive:
                to_remove.append(car)
                continue
            car.y += drift
            # Car passed off bottom
            if car.y >= self._HEIGHT:
                car.alive = False
                self._cars_passed += 1
                self._on_point_scored(1)
                reward += 1.0
                to_remove.append(car)
            # Collision
            elif (
                car.y >= self._PLAYER_ROW - 1
                and car.y <= self._PLAYER_ROW
                and car.data["lane"] == self._player_lane
            ):
                car.alive = False
                to_remove.append(car)
                self._speed = 1
                self._on_life_lost()
                self._message = "Crash! Speed reset."
                if self._game_over:
                    return -1.0, True, info

        for c in to_remove:
            if c in self._traffic:
                self._traffic.remove(c)

        # Day progression
        if self._cars_passed >= self._CARS_TARGET * self._day:
            self._day += 1
            self._message = f"Day {self._day}! Keep going!"

        info["cars_passed"] = self._cars_passed
        info["speed"] = self._speed
        info["day"] = self._day
        self._redraw()
        return reward, False, info

    def _redraw(self) -> None:
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                self._set_cell(x, y, " ")

        # Road borders
        road_r = (
            self._ROAD_LEFT
            + 1
            + self._NUM_LANES * self._LANE_WIDTH
        )
        for y in range(self._HEIGHT):
            self._set_cell(self._ROAD_LEFT, y, "│")
            if road_r < self._WIDTH:
                self._set_cell(road_r, y, "│")

        # Lane dividers (dashed)
        for lane in range(1, self._NUM_LANES):
            lx = self._ROAD_LEFT + 1 + lane * self._LANE_WIDTH
            for y in range(self._HEIGHT):
                if y % 3 == 0:
                    self._set_cell(lx, y, ":")

        # Shoulder/grass
        for y in range(self._HEIGHT):
            for x in range(self._ROAD_LEFT):
                self._set_cell(x, y, "~")
            for x in range(road_r + 1, self._WIDTH):
                self._set_cell(x, y, "~")

        # Traffic cars
        for car in self._traffic:
            if car.alive and 0 <= car.y < self._HEIGHT:
                self._set_cell(car.x, car.y, "C")

    def _advance_entities(self) -> None:
        # Handled in _game_step
        pass

    def _render_current_observation(self, **kw: Any):  # type: ignore[override]
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
        target = self._CARS_TARGET * self._day
        hud = (
            f"Score: {self._score}  "
            f"Cars: {self._cars_passed}  "
            f"Speed: {self._speed}  "
            f"Lives: {self._lives}  "
            f"Day: {self._day}\n"
            f"Target: {target}/day"
        )
        return GridObservation(
            grid=grid_to_string(render),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "│": "road edge",
            ":": "lane divider",
            "~": "grass",
            "C": "traffic car",
            " ": "road",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Race down the highway. Steer LEFT/RIGHT to "
            "change lanes. ACCELERATE to go faster, "
            "BRAKE to slow down. Pass cars to score. "
            "Avoid collisions or lose a life."
        )
