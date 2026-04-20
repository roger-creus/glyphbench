"""Atari River Raid environment.

Vertical scrolling river shooter with fuel management.

Gym ID: atlas_rl/atari-riverraid-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec

from .base import AtariBase, AtariEntity


class RiverRaidEnv(AtariBase):
    """River Raid: fly up a scrolling river, shoot enemies, manage fuel.

    20x24 viewport. River scrolls down; enemies, fuel depots appear.
    Running out of fuel or crashing ends the game.

    Actions: NOOP, UP, DOWN, LEFT, RIGHT, FIRE
    Reward: +10 enemy, +20 fuel depot
    Lives: 3
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "accelerate (scroll faster)",
            "decelerate",
            "move left",
            "move right",
            "fire a bullet forward",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 24
    _PLAYER_Y = 20
    _RIVER_LEFT = 4
    _RIVER_RIGHT = 15
    _MAX_FUEL = 100
    _MAX_BULLETS = 2

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._bullets: list[AtariEntity] = []
        self._obstacles: list[AtariEntity] = []
        self._fuel: int = self._MAX_FUEL
        self._scroll_timer: int = 0
        self._scroll_speed: int = 2
        self._step_counter: int = 0
        self._river_l: list[int] = []
        self._river_r: list[int] = []

    def env_id(self) -> str:
        return "atlas_rl/atari-riverraid-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._bullets = []
        self._obstacles = []
        self._step_counter = 0
        self._scroll_timer = 0
        self._scroll_speed = 2
        self._fuel = self._MAX_FUEL

        # Init river banks
        self._river_l = [self._RIVER_LEFT] * self._HEIGHT
        self._river_r = [self._RIVER_RIGHT] * self._HEIGHT

        self._player_x = self._WIDTH // 2
        self._player_y = self._PLAYER_Y

        # Spawn some initial enemies
        rng = self.rng
        for row in range(2, self._HEIGHT - 5, 3):
            if int(rng.integers(0, 3)) > 0:
                ex = int(rng.integers(
                    self._river_l[row] + 1,
                    self._river_r[row]
                ))
                etype = "ship" if int(rng.integers(0, 2)) == 0 else "heli"
                ch = "S" if etype == "ship" else "H"
                e = self._add_entity(etype, ch, ex, row)
                e.data["dx"] = 1 if int(rng.integers(0, 2)) == 0 else -1
                e.data["timer"] = 0
                self._obstacles.append(e)
            # Occasionally place fuel depot
            if int(rng.integers(0, 5)) == 0:
                fx = int(rng.integers(
                    self._river_l[row] + 1,
                    self._river_r[row]
                ))
                f = self._add_entity("fuel", "F", fx, row)
                self._obstacles.append(f)

        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        # Move player
        if action_name == "LEFT" and self._player_x > 1:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._player_x < self._WIDTH - 2:
            self._player_x += 1
            self._player_dir = (1, 0)
        elif action_name == "UP":
            self._scroll_speed = min(3, self._scroll_speed + 1)
        elif action_name == "DOWN":
            self._scroll_speed = max(1, self._scroll_speed - 1)

        # Fire
        if action_name == "FIRE" and len(self._bullets) < self._MAX_BULLETS:
            b = self._add_entity(
                "bullet", "!", self._player_x,
                self._player_y - 1, dy=-1
            )
            self._bullets.append(b)

        # Scroll (move obstacles down)
        self._scroll_timer += 1
        do_scroll = self._scroll_timer >= (4 - self._scroll_speed)
        if do_scroll:
            self._scroll_timer = 0
            for obs in self._obstacles:
                if obs.alive:
                    obs.y += 1
                    if obs.y >= self._HEIGHT - 1:
                        obs.alive = False

            # Randomly spawn new obstacles at top
            rng = self.rng
            rl, rr = self._river_l[2], self._river_r[2]
            if int(rng.integers(0, 3)) > 0 and rr > rl + 1:
                ex = int(rng.integers(rl + 1, rr))
                roll = int(rng.integers(0, 6))
                if roll < 4:
                    nm = "ship" if roll < 2 else "heli"
                    ch = "S" if roll < 2 else "H"
                    e = self._add_entity(nm, ch, ex, 1)
                    d = 1 if int(rng.integers(0, 2)) == 0 else -1
                    e.data.update(dx=d, timer=0)
                    self._obstacles.append(e)
                elif roll == 4:
                    f = self._add_entity("fuel", "F", ex, 1)
                    self._obstacles.append(f)

        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.y += b.dy
            if b.y <= 0:
                b.alive = False
                continue
            for obs in self._obstacles:
                if not obs.alive:
                    continue
                if obs.x == b.x and obs.y == b.y:
                    b.alive = False
                    if obs.etype == "fuel":
                        obs.alive = False
                        self._on_point_scored(20)
                        reward += 20
                        self._fuel = min(
                            self._MAX_FUEL, self._fuel + 30
                        )
                        self._message = "Fuel! +20 & refuel"
                    else:
                        obs.alive = False
                        self._on_point_scored(10)
                        reward += 10
                        self._message = "Enemy destroyed! +10"
                    break
        self._bullets = [b for b in self._bullets if b.alive]

        # Move enemies horizontally
        for obs in self._obstacles:
            if not obs.alive:
                continue
            if obs.etype in ("ship", "heli"):
                obs.data["timer"] = obs.data.get("timer", 0) + 1
                if obs.data["timer"] % 3 == 0:
                    obs.x += obs.data.get("dx", 0)
                    rl = self._river_l[min(obs.y, self._HEIGHT - 1)]
                    rr = self._river_r[min(obs.y, self._HEIGHT - 1)]
                    if obs.x <= rl or obs.x >= rr:
                        obs.data["dx"] = -obs.data.get("dx", 1)

        # Player picks up fuel by touching
        for obs in self._obstacles:
            if (
                obs.alive
                and obs.etype == "fuel"
                and obs.x == self._player_x
                and obs.y == self._player_y
            ):
                obs.alive = False
                self._fuel = min(self._MAX_FUEL, self._fuel + 30)
                self._on_point_scored(20)
                reward += 20
                self._message = "Fuel pickup! +20"

        # Player collision with enemies
        for obs in self._obstacles:
            if (
                obs.alive
                and obs.etype in ("ship", "heli")
                and obs.x == self._player_x
                and obs.y == self._player_y
            ):
                obs.alive = False
                self._on_life_lost()
                self._message = "Crash! Lost a life."
                self._player_x = self._WIDTH // 2
                break

        # Bank collision
        rl = self._river_l[self._player_y]
        rr = self._river_r[self._player_y]
        if self._player_x <= rl or self._player_x >= rr:
            self._on_life_lost()
            self._message = "Hit the bank! Lost a life."
            self._player_x = self._WIDTH // 2

        # Fuel consumption
        self._fuel -= 1
        if self._fuel <= 0:
            self._game_over = True
            self._message = "Out of fuel! Game Over!"

        self._obstacles = [o for o in self._obstacles if o.alive]
        self._redraw()
        info["fuel"] = self._fuel
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(self._HEIGHT):
            for x in range(self._WIDTH):
                rl = self._river_l[y]
                rr = self._river_r[y]
                if y == 0 or y == self._HEIGHT - 1:
                    self._set_cell(x, y, "-")
                elif x <= rl or x >= rr:
                    self._set_cell(x, y, "#")
                else:
                    self._set_cell(x, y, "~")

        for obs in self._obstacles:
            if obs.alive:
                self._set_cell(obs.x, obs.y, obs.char)
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, "!")

    def _render_current_observation(self) -> Any:
        """Override to add fuel and enemy count to HUD."""
        from atlas_rl.core.observation import GridObservation as GO

        obs = super()._render_current_observation()
        enemies = sum(
            1 for o in self._obstacles
            if o.alive and o.etype in ("ship", "heli")
        )
        hud = (
            f"Score: {self._score}    Lives: {self._lives}"
            f"    Level: {self._level}    "
            f"Fuel: {self._fuel}\n"
            f"Enemies: {enemies}"
        )
        return GO(
            grid=obs.grid, legend=obs.legend,
            hud=hud, message=obs.message,
        )

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "-": "border",
            "#": "river bank (deadly)",
            "~": "river (safe)",
            "S": "enemy ship (10pts)",
            "H": "enemy helicopter (10pts)",
            "F": "fuel depot (+30 fuel, 20pts)",
            "!": "your bullet",
            " ": "empty",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Fly up the river, shoot enemies and collect fuel. "
            "Don't crash into banks or enemies. "
            "Running out of fuel ends the game."
        )
