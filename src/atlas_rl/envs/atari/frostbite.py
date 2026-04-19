"""Atari Frostbite environment.

Ice floe jumping game. Jump between rows of moving ice floes
to build an igloo piece by piece.

Gym ID: atlas_rl/atari-frostbite-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec

from .base import AtariBase, AtariEntity


class FrostbiteEnv(AtariBase):
    """Frostbite: ice floe jumping game.

    Rows of ice floes move horizontally. The player jumps between
    them. Each floe landed on adds a section to the igloo.
    Complete the igloo to advance.

    Grid: 20 wide x 16 tall.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "RIGHT", "LEFT", "DOWN"),
        descriptions=(
            "do nothing",
            "jump up one row",
            "move right",
            "move left",
            "jump down one row",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 16
    _NUM_FLOE_ROWS = 4
    _IGLOO_SECTIONS = 10
    _SHORE_Y = 2  # top shore row (igloo area)
    _WATER_START = 4  # first water/floe row
    _BOTTOM_SHORE = 14  # bottom shore

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._lives = 3
        self._igloo_built: int = 0
        self._floe_rows: list[list[AtariEntity]] = []
        self._on_floe: bool = False
        self._current_floe: AtariEntity | None = None
        self._visited_floes: set[int] = set()
        self._temperature: int = 45  # decreasing timer

    def env_id(self) -> str:
        return "atlas_rl/atari-frostbite-v0"

    def _generate_level(self, seed: int) -> None:
        self._lives = 3
        self._igloo_built = 0
        self._on_floe = False
        self._current_floe = None
        self._visited_floes = set()
        self._temperature = 45
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._floe_rows = []

        # Border
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "-")
            self._set_cell(x, self._HEIGHT - 1, "-")

        # Top shore (igloo area)
        for x in range(self._WIDTH):
            for y in range(1, self._WATER_START):
                self._set_cell(x, y, ".")

        # Water rows
        for y in range(self._WATER_START, self._BOTTOM_SHORE):
            for x in range(self._WIDTH):
                self._set_cell(x, y, "~")

        # Bottom shore
        for x in range(self._WIDTH):
            self._set_cell(x, self._BOTTOM_SHORE, "=")

        # Create ice floe rows
        row_ys = [
            self._WATER_START + 1,
            self._WATER_START + 3,
            self._WATER_START + 5,
            self._WATER_START + 7,
        ]
        for i, ry in enumerate(row_ys):
            if ry >= self._BOTTOM_SHORE:
                continue
            direction = 1 if i % 2 == 0 else -1
            row_floes: list[AtariEntity] = []
            # Place 2-3 floes per row
            num_floes = 2 + (i % 2)
            spacing = self._WIDTH // (num_floes + 1)
            for j in range(num_floes):
                fx = (spacing * (j + 1)) % self._WIDTH
                floe = self._add_entity(
                    "floe", "I", fx, ry, dx=direction, dy=0
                )
                floe.data["row"] = i
                floe.data["width"] = 3
                row_floes.append(floe)
            self._floe_rows.append(row_floes)

        # Player starts on bottom shore
        self._player_x = self._WIDTH // 2
        self._player_y = self._BOTTOM_SHORE

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        self._temperature -= 1
        if self._temperature <= 0:
            self._on_life_lost()
            self._temperature = 45
            self._player_x = self._WIDTH // 2
            self._player_y = self._BOTTOM_SHORE
            self._on_floe = False
            self._current_floe = None
            self._message = "Froze! Lost a life."

        dx, dy = 0, 0

        if action_name == "LEFT":
            dx = -1
        elif action_name == "RIGHT":
            dx = 1
        elif action_name == "UP":
            dy = -2  # jump up to previous row
        elif action_name == "DOWN":
            dy = 2  # jump down to next row

        # Move horizontally
        new_x = self._player_x + dx
        if new_x < 0:
            new_x = 0
        elif new_x >= self._WIDTH:
            new_x = self._WIDTH - 1
        self._player_x = new_x

        # Vertical jump
        if dy != 0:
            new_y = self._player_y + dy
            if 0 < new_y < self._HEIGHT - 1:
                self._player_y = new_y
                self._on_floe = False
                self._current_floe = None

        # If riding a floe, move with it
        if self._on_floe and self._current_floe is not None:
            if self._current_floe.alive:
                self._player_x = (self._player_x + self._current_floe.dx) % self._WIDTH
            else:
                self._on_floe = False
                self._current_floe = None

        # Move floes
        for row in self._floe_rows:
            for floe in row:
                if floe.alive:
                    floe.x = (floe.x + floe.dx) % self._WIDTH

        # Check if player is on a floe
        if not self._on_shore():
            landed = False
            for row in self._floe_rows:
                for floe in row:
                    if floe.alive and floe.y == self._player_y:
                        fw = floe.data.get("width", 3)
                        # Check if player is within floe width (with wrapping)
                        for offset in range(fw):
                            floe_cell = (floe.x + offset) % self._WIDTH
                            if floe_cell == self._player_x:
                                landed = True
                                if not self._on_floe or self._current_floe is not floe:
                                    self._on_floe = True
                                    self._current_floe = floe
                                    # Score for landing on new floe
                                    floe_id = id(floe)
                                    if floe_id not in self._visited_floes:
                                        self._visited_floes.add(floe_id)
                                        self._on_point_scored(10)
                                        reward += 10
                                        self._igloo_built += 1
                                        self._message = f"Floe! +10 (igloo {self._igloo_built}/{self._IGLOO_SECTIONS})"
                                break
                    if landed:
                        break
                if landed:
                    break

            if not landed:
                self._on_floe = False
                self._current_floe = None
                # In water - lose a life
                if self._is_water(self._player_x, self._player_y):
                    self._on_life_lost()
                    self._player_x = self._WIDTH // 2
                    self._player_y = self._BOTTOM_SHORE
                    self._on_floe = False
                    self._current_floe = None
                    self._message = "Fell in water!"

        # Check if on top shore and igloo complete
        if self._player_y <= self._SHORE_Y + 1 and self._on_shore() and self._igloo_built >= self._IGLOO_SECTIONS:
                self._on_point_scored(100)
                reward += 100
                self._message = "Igloo complete! +100"
                self._level += 1
                self._igloo_built = 0
                self._visited_floes = set()
                self._temperature = 45
                self._player_x = self._WIDTH // 2
                self._player_y = self._BOTTOM_SHORE
                self._on_floe = False
                self._current_floe = None
                # Rebuild for new level
                self._generate_level(self._level)

        # Redraw floes on the grid
        self._redraw_water()

        info["igloo"] = self._igloo_built
        info["temperature"] = self._temperature

        return reward, self._game_over, info

    def _on_shore(self) -> bool:
        return self._player_y <= self._SHORE_Y + 1 or self._player_y >= self._BOTTOM_SHORE

    def _is_water(self, x: int, y: int) -> bool:
        return self._WATER_START <= y < self._BOTTOM_SHORE

    def _redraw_water(self) -> None:
        """Redraw water area and floes."""
        for y in range(self._WATER_START, self._BOTTOM_SHORE):
            for x in range(self._WIDTH):
                self._set_cell(x, y, "~")

        # Draw floes
        for row in self._floe_rows:
            for floe in row:
                if floe.alive:
                    fw = floe.data.get("width", 3)
                    for offset in range(fw):
                        fx = (floe.x + offset) % self._WIDTH
                        if 0 <= fx < self._WIDTH:
                            self._set_cell(fx, floe.y, "I")

        # Draw igloo progress on top shore
        igloo_x = self._WIDTH // 2 - 2
        for i in range(min(self._igloo_built, self._IGLOO_SECTIONS)):
            ix = igloo_x + (i % 5)
            iy = 1 + (i // 5)
            if 0 < ix < self._WIDTH and 0 < iy < self._WATER_START:
                self._set_cell(ix, iy, "O")

    def _advance_entities(self) -> None:
        # Floes are moved in _game_step; skip default entity advancement
        # to avoid double-moving and out-of-bounds killing
        pass

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "-": "border",
            ".": "shore",
            "~": "water",
            "=": "shore/ground",
            "I": "ice floe",
            "O": "igloo section",
            " ": "empty",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Jump between ice floes to build the igloo. "
            "Land on each floe to add a section. "
            "Complete the igloo then reach the top shore. "
            "Don't fall in the water or freeze."
        )
