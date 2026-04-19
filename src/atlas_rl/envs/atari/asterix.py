"""Atari Asterix environment.

Collect food items while avoiding enemies in horizontal lanes.

Gym ID: atlas_rl/atari-asterix-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec

from .base import AtariBase, AtariEntity


class AsterixEnv(AtariBase):
    """Asterix: collect food, dodge enemies in lanes.

    20x16 grid. Food and enemies scroll horizontally.
    Move up/down between lanes to grab food and dodge enemies.

    Actions: NOOP, UP, DOWN
    Reward: +10 per food collected, -1 per food missed
    Lives: 3 (lost on enemy collision)
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN"),
        descriptions=(
            "do nothing",
            "move up one lane",
            "move down one lane",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 16
    _NUM_LANES = 8
    _LANE_START = 4
    _PLAYER_X = 2

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._food: list[AtariEntity] = []
        self._enemies: list[AtariEntity] = []
        self._step_counter: int = 0
        self._spawn_rate: int = 8

    def env_id(self) -> str:
        return "atlas_rl/atari-asterix-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._food = []
        self._enemies = []
        self._step_counter = 0
        self._spawn_rate = max(4, 8 - self._level)

        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "-")
            self._set_cell(x, self._HEIGHT - 1, "-")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "|")
            self._set_cell(self._WIDTH - 1, y, "|")

        # Lane separators
        for x in range(1, self._WIDTH - 1):
            self._set_cell(x, self._LANE_START - 1, ".")

        # Player starts in middle lane
        self._player_x = self._PLAYER_X
        self._player_y = self._LANE_START + self._NUM_LANES // 2

    def _lane_y(self, lane: int) -> int:
        return self._LANE_START + lane

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        # Move player
        if action_name == "UP":
            ny = self._player_y - 1
            if ny >= self._LANE_START:
                self._player_y = ny
        elif action_name == "DOWN":
            ny = self._player_y + 1
            if ny < self._LANE_START + self._NUM_LANES:
                self._player_y = ny

        # Spawn food and enemies
        if self._step_counter % self._spawn_rate == 0:
            rng = self.rng
            lane = int(rng.integers(self._NUM_LANES))
            y = self._lane_y(lane)
            if rng.random() < 0.6:
                f = self._add_entity(
                    "food", "*", self._WIDTH - 2, y, dx=-1
                )
                f.data["timer"] = 0
                self._food.append(f)
            else:
                e = self._add_entity(
                    "enemy", "E", self._WIDTH - 2, y, dx=-1
                )
                e.data["timer"] = 0
                self._enemies.append(e)

        # Move food
        for f in self._food:
            if not f.alive:
                continue
            f.data["timer"] += 1
            if f.data["timer"] >= 2:
                f.data["timer"] = 0
                f.x += f.dx
                if f.x <= 0:
                    f.alive = False

        # Move enemies
        speed = min(2, 1 + self._level // 3)
        for e in self._enemies:
            if not e.alive:
                continue
            e.data["timer"] += 1
            if e.data["timer"] >= max(1, 3 - speed):
                e.data["timer"] = 0
                e.x += e.dx
                if e.x <= 0:
                    e.alive = False

        # Check food collection
        for f in self._food:
            if (
                f.alive
                and f.x == self._player_x
                and f.y == self._player_y
            ):
                f.alive = False
                self._on_point_scored(10)
                reward += 10
                self._message = "Yum! +10"

        # Check enemy collision
        for e in self._enemies:
            if (
                e.alive
                and e.x == self._player_x
                and e.y == self._player_y
            ):
                e.alive = False
                self._on_life_lost()
                self._message = "Hit by enemy!"
                self._player_y = (
                    self._LANE_START + self._NUM_LANES // 2
                )

        # Level up every 100 points
        if (
            self._score > 0
            and self._score % 100 == 0
            and self._step_counter % self._spawn_rate == 0
        ):
            self._level += 1
            self._spawn_rate = max(4, 8 - self._level)

        # Cleanup
        self._food = [f for f in self._food if f.alive]
        self._enemies = [e for e in self._enemies if e.alive]

        self._redraw()
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(self._LANE_START, self._LANE_START + self._NUM_LANES):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")
        for f in self._food:
            if f.alive:
                self._set_cell(f.x, f.y, f.char)
        for e in self._enemies:
            if e.alive:
                self._set_cell(e.x, e.y, e.char)

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "-": "wall",
            "|": "wall",
            ".": "lane border",
            "*": "food (+10)",
            "E": "enemy (dodge!)",
            " ": "empty",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Collect food (*) by moving into its lane. "
            "Avoid enemies (E) or lose a life. "
            "Food and enemies scroll from right to left."
        )
