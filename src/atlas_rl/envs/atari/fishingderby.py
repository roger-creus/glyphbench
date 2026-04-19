"""Atari Fishing Derby environment.

Two players fishing. Catch fish to score.

Gym ID: atlas_rl/atari-fishingderby-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec

from .base import AtariBase, AtariEntity


class FishingDerbyEnv(AtariBase):
    """Fishing Derby: catch fish from a lake.

    20x16 grid. Player and AI each have a fishing line.
    Move hook to catch fish, reel them in.

    Actions: NOOP, UP, DOWN, LEFT, RIGHT, REEL
    Reward: +1 per fish caught
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "UP", "DOWN", "LEFT",
            "RIGHT", "REEL",
        ),
        descriptions=(
            "do nothing",
            "move hook up",
            "move hook down",
            "move hook left",
            "move hook right",
            "reel in the line",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 16
    _WATER_TOP = 4
    _WATER_BOT = 14
    _PIER_Y = 3
    _MAX_FISH = 6
    _GAME_TIME = 500

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._hook_x: int = 0
        self._hook_y: int = 0
        self._opp_hook_x: int = 0
        self._opp_hook_y: int = 0
        self._fish: list[AtariEntity] = []
        self._hooked_fish: AtariEntity | None = None
        self._opp_hooked: AtariEntity | None = None
        self._opp_score: int = 0
        self._timer: int = 0
        self._reeling: bool = False

    def env_id(self) -> str:
        return "atlas_rl/atari-fishingderby-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._fish = []
        self._hooked_fish = None
        self._opp_hooked = None
        self._opp_score = 0
        self._timer = 0
        self._reeling = False
        self._lives = 99

        # Player on left pier
        self._player_x = 3
        self._player_y = self._PIER_Y
        self._hook_x = 5
        self._hook_y = self._WATER_TOP + 2

        # Opponent on right pier
        self._opp_hook_x = 14
        self._opp_hook_y = self._WATER_TOP + 2

        # Spawn fish
        rng = self.rng
        for _i in range(self._MAX_FISH):
            fx = int(
                rng.integers(2, self._WIDTH - 2)
            )
            fy = int(
                rng.integers(
                    self._WATER_TOP + 1,
                    self._WATER_BOT,
                )
            )
            dx = 1 if rng.random() < 0.5 else -1
            fish = self._add_entity(
                "fish", "F", fx, fy, dx=dx
            )
            fish.data["speed_timer"] = 0
            self._fish.append(fish)

        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._timer += 1

        # Move hook
        if self._hooked_fish is None:
            if action_name == "UP":
                ny = self._hook_y - 1
                if ny >= self._WATER_TOP:
                    self._hook_y = ny
            elif action_name == "DOWN":
                ny = self._hook_y + 1
                if ny < self._WATER_BOT:
                    self._hook_y = ny
            elif action_name == "LEFT":
                nx = self._hook_x - 1
                if nx >= 1:
                    self._hook_x = nx
            elif action_name == "RIGHT":
                nx = self._hook_x + 1
                if nx < self._WIDTH - 1:
                    self._hook_x = nx

        # Check if hook catches a fish
        if self._hooked_fish is None:
            for fish in self._fish:
                if not fish.alive:
                    continue
                if (
                    fish.x == self._hook_x
                    and fish.y == self._hook_y
                ):
                    self._hooked_fish = fish
                    self._message = "Fish on the hook!"
                    break

        # Reel in
        if action_name == "REEL" and self._hooked_fish:
            self._hook_y -= 1
            self._hooked_fish.x = self._hook_x
            self._hooked_fish.y = self._hook_y
            if self._hook_y <= self._WATER_TOP:
                self._hooked_fish.alive = False
                self._fish.remove(self._hooked_fish)
                self._hooked_fish = None
                self._on_point_scored(1)
                reward = 1.0
                self._message = "Caught a fish! +1"
                self._hook_y = self._WATER_TOP + 2
                self._maybe_spawn_fish()

        # Move fish
        for fish in self._fish:
            if not fish.alive or fish is self._hooked_fish:
                continue
            fish.data["speed_timer"] += 1
            if fish.data["speed_timer"] >= 3:
                fish.data["speed_timer"] = 0
                fish.x += fish.dx
                if (
                    fish.x <= 1
                    or fish.x >= self._WIDTH - 2
                ):
                    fish.dx = -fish.dx
                    fish.x += fish.dx

        # Opponent AI
        self._opponent_ai()

        # Time check
        terminated = False
        if self._timer >= self._GAME_TIME:
            terminated = True
            if self._score > self._opp_score:
                self._message = "You win!"
            elif self._opp_score > self._score:
                self._message = "Opponent wins!"
            else:
                self._message = "Tie!"

        info["opp_score"] = self._opp_score
        info["time_left"] = self._GAME_TIME - self._timer
        self._redraw()
        return reward, terminated, info

    def _maybe_spawn_fish(self) -> None:
        if len(self._fish) < self._MAX_FISH:
            rng = self.rng
            fx = int(rng.integers(2, self._WIDTH - 2))
            fy = int(
                rng.integers(
                    self._WATER_TOP + 1,
                    self._WATER_BOT,
                )
            )
            dx = 1 if rng.random() < 0.5 else -1
            fish = self._add_entity(
                "fish", "F", fx, fy, dx=dx
            )
            fish.data["speed_timer"] = 0
            self._fish.append(fish)

    def _opponent_ai(self) -> None:
        rng = self.rng
        # Opponent tries to catch fish
        if self._opp_hooked is None:
            # Find nearest fish
            best = None
            best_dist = 999
            for fish in self._fish:
                if not fish.alive:
                    continue
                if fish is self._hooked_fish:
                    continue
                d = (
                    abs(fish.x - self._opp_hook_x)
                    + abs(fish.y - self._opp_hook_y)
                )
                if d < best_dist:
                    best_dist = d
                    best = fish
            if best and rng.random() < 0.5:
                if self._opp_hook_x < best.x:
                    self._opp_hook_x += 1
                elif self._opp_hook_x > best.x:
                    self._opp_hook_x -= 1
                if self._opp_hook_y < best.y:
                    self._opp_hook_y += 1
                elif self._opp_hook_y > best.y:
                    self._opp_hook_y -= 1
            # Check catch
            for fish in self._fish:
                if not fish.alive:
                    continue
                if fish is self._hooked_fish:
                    continue
                if (
                    fish.x == self._opp_hook_x
                    and fish.y == self._opp_hook_y
                ):
                    self._opp_hooked = fish
                    break
        else:
            # Reel in
            self._opp_hook_y -= 1
            self._opp_hooked.x = self._opp_hook_x
            self._opp_hooked.y = self._opp_hook_y
            if self._opp_hook_y <= self._WATER_TOP:
                self._opp_hooked.alive = False
                if self._opp_hooked in self._fish:
                    self._fish.remove(self._opp_hooked)
                self._opp_hooked = None
                self._opp_score += 1
                self._opp_hook_y = self._WATER_TOP + 2
                self._maybe_spawn_fish()

    def _redraw(self) -> None:
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                self._set_cell(x, y, " ")

        # Sky
        for y in range(self._PIER_Y):
            for x in range(self._WIDTH):
                self._set_cell(x, y, " ")

        # Pier
        for x in range(self._WIDTH):
            self._set_cell(x, self._PIER_Y, "=")

        # Water surface
        for x in range(self._WIDTH):
            self._set_cell(x, self._WATER_TOP, "~")

        # Water
        for y in range(
            self._WATER_TOP + 1, self._WATER_BOT + 1
        ):
            for x in range(self._WIDTH):
                self._set_cell(x, y, ".")

        # Bottom
        for x in range(self._WIDTH):
            self._set_cell(x, self._WATER_BOT, "-")

        # Player hook and line
        self._set_cell(self._hook_x, self._hook_y, "J")
        for y in range(self._PIER_Y + 1, self._hook_y):
            self._set_cell(self._hook_x, y, ":")

        # Opponent hook
        self._set_cell(
            self._opp_hook_x, self._opp_hook_y, "j"
        )

        # Opponent position on pier
        self._set_cell(16, self._PIER_Y - 1, "V")

        # Fish
        for fish in self._fish:
            if fish.alive:
                self._set_cell(fish.x, fish.y, "F")

    def _advance_entities(self) -> None:
        pass

    def _render_current_observation(self, **kw: Any):  # type: ignore[override]
        from atlas_rl.core.ascii_primitives import (
            build_legend,
            grid_to_string,
        )
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
        left = self._GAME_TIME - self._timer
        hud = (
            f"You: {self._score}  "
            f"Opp: {self._opp_score}  "
            f"Time: {left}"
        )
        return GridObservation(
            grid=grid_to_string(render),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "=": "pier",
            "~": "water surface",
            ".": "water",
            "-": "lake bottom",
            "J": "your hook",
            "j": "opponent hook",
            ":": "fishing line",
            "F": "fish",
            "V": "opponent",
            " ": "sky",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Fishing derby! Move your hook with "
            "UP/DOWN/LEFT/RIGHT to reach a fish, "
            "then REEL to pull it up. "
            "Catch more fish than your opponent."
        )
