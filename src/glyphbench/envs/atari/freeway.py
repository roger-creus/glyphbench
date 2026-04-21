"""Atari Freeway environment.

Guide a chicken across a highway full of traffic.

Gym ID: glyphbench/atari-freeway-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class FreewayEnv(AtariBase):
    """Freeway: guide a chicken across lanes of traffic.

    20x16 grid. Cars move horizontally at various speeds.
    Reach the top to score, then restart at the bottom.

    Actions: NOOP, UP, DOWN
    Reward: +1 per successful crossing
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN"),
        descriptions=(
            "do nothing",
            "move up one row",
            "move down one row",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 16
    _PLAYER_START_Y = 14
    _GOAL_Y = 1
    _NUM_LANES = 12

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._lanes: list[list[AtariEntity]] = []

    def env_id(self) -> str:
        return "glyphbench/atari-freeway-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._lanes = []

        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")

        # Goal area (top)
        for x in range(self._WIDTH):
            self._set_cell(x, self._GOAL_Y, "·")

        # Start area (bottom)
        for x in range(self._WIDTH):
            self._set_cell(x, self._PLAYER_START_Y, "·")

        # Create lanes with cars
        rng = self.rng
        for lane_idx in range(self._NUM_LANES):
            lane_y = 2 + lane_idx
            if lane_y >= self._PLAYER_START_Y:
                break

            direction = 1 if lane_idx % 2 == 0 else -1
            # Speed increases with level; at least 1
            speed = 1 + (lane_idx % 3)
            num_cars = 2 + (lane_idx % 2)

            lane_cars: list[AtariEntity] = []
            spacing = self._WIDTH // (num_cars + 1)
            for car_idx in range(num_cars):
                cx = (spacing * (car_idx + 1) + int(rng.integers(0, 3))) % self._WIDTH
                # Cars are 2-3 chars wide; we represent with single char
                car = self._add_entity("car", "C", cx, lane_y, dx=direction)
                car.data["speed"] = speed
                car.data["timer"] = 0
                lane_cars.append(car)
            self._lanes.append(lane_cars)

        # Player at bottom center
        self._player_x = self._WIDTH // 2
        self._player_y = self._PLAYER_START_Y

        self._redraw_lanes()

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Move player
        if action_name == "UP":
            new_y = self._player_y - 1
            if new_y >= self._GOAL_Y:
                self._player_y = new_y
                self._player_dir = (0, -1)
        elif action_name == "DOWN":
            new_y = self._player_y + 1
            if new_y <= self._PLAYER_START_Y:
                self._player_y = new_y
                self._player_dir = (0, 1)

        # Move cars
        for lane in self._lanes:
            for car in lane:
                if not car.alive:
                    continue
                spd = car.data.get("speed", 1)
                car.data["timer"] = car.data.get("timer", 0) + 1
                if car.data["timer"] >= (4 - min(spd, 3)):
                    car.data["timer"] = 0
                    car.x = (car.x + car.dx) % self._WIDTH

        # Check collision with cars
        hit = False
        for lane in self._lanes:
            for car in lane:
                if car.alive and car.x == self._player_x and car.y == self._player_y:
                    hit = True
                    break
            if hit:
                break

        if hit:
            # Push player back to start (no life lost in Freeway)
            self._player_y = self._PLAYER_START_Y
            self._message = "Hit by car! Back to start."

        # Check goal
        if self._player_y <= self._GOAL_Y:
            self._on_point_scored(1)
            reward += 1
            self._message = "Crossed! +1"
            self._player_y = self._PLAYER_START_Y

        self._redraw_lanes()
        info["crossings"] = self._score
        return reward, False, info  # Freeway never terminates (time-limited)

    def _redraw_lanes(self) -> None:
        """Redraw the lane area."""
        # Clear lane area
        for y in range(2, self._PLAYER_START_Y):
            for x in range(self._WIDTH):
                self._set_cell(x, y, " ")

        # Draw cars
        for lane in self._lanes:
            for car in lane:
                if car.alive and 0 <= car.x < self._WIDTH:
                    self._set_cell(car.x, car.y, "C")

    def _advance_entities(self) -> None:
        # Cars are moved in _game_step
        pass

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "border",
            "·": "safe zone",
            "C": "car",
            " ": "road",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        dirs: list[str] = []
        for lane in self._lanes:
            if lane:
                d = lane[0].dx
                dirs.append("→" if d == 1 else "←")
        extra = f"Lane car dirs: {' '.join(dirs)}"
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Guide the chicken across the highway. "
            "Press UP to move toward the top. "
            "Avoid cars or you'll be sent back to the start. "
            "Each successful crossing scores +1."
        )
