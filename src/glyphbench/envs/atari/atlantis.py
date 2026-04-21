"""Atari Atlantis environment.

Three fixed turrets defending a city from flying enemies.

Gym ID: glyphbench/atari-atlantis-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity


class AtlantisEnv(AtariBase):
    """Atlantis: defend your city with three turrets.

    30x16 grid. Flying enemies cross the screen. Three fixed
    turrets fire at angles to intercept. City buildings can be
    destroyed.

    Actions: NOOP, FIRE_LEFT, FIRE_CENTER, FIRE_RIGHT
    Reward: +20 per enemy destroyed
    Lives: lost when all buildings destroyed
    """

    action_spec = ActionSpec(
        names=("NOOP", "FIRE_LEFT", "FIRE_CENTER", "FIRE_RIGHT"),
        descriptions=(
            "do nothing",
            "fire from left turret (diagonal right)",
            "fire from center turret (straight up)",
            "fire from right turret (diagonal left)",
        ),
    )

    _WIDTH = 30
    _HEIGHT = 16
    _GROUND_Y = 13
    _TURRET_LEFT = 3
    _TURRET_CENTER = 15
    _TURRET_RIGHT = 27

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._flyers: list[AtariEntity] = []
        self._bullets: list[AtariEntity] = []
        self._buildings: list[tuple[int, int]] = []
        self._step_counter: int = 0
        self._spawn_interval: int = 10

    def env_id(self) -> str:
        return "glyphbench/atari-atlantis-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._flyers = []
        self._bullets = []
        self._step_counter = 0
        self._spawn_interval = max(5, 12 - self._level)

        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")

        # Ground and buildings
        self._buildings = []
        for x in range(self._WIDTH):
            self._set_cell(x, self._GROUND_Y, "=")

        building_xs = [7, 10, 13, 17, 20, 23]
        for bx in building_xs:
            for dy in range(2):
                pos = (bx, self._GROUND_Y - 1 - dy)
                self._buildings.append(pos)

        # Turrets (always present)
        self._set_cell(self._TURRET_LEFT, self._GROUND_Y - 1, "T")
        self._set_cell(
            self._TURRET_CENTER, self._GROUND_Y - 1, "T"
        )
        self._set_cell(
            self._TURRET_RIGHT, self._GROUND_Y - 1, "T"
        )

        # Player position (center turret for display)
        self._player_x = self._TURRET_CENTER
        self._player_y = self._GROUND_Y - 1

        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        # Fire from turrets
        if action_name == "FIRE_LEFT" and len(self._bullets) < 4:
            b = self._add_entity(
                "bullet", "/", self._TURRET_LEFT,
                self._GROUND_Y - 2, dx=1, dy=-1,
            )
            self._bullets.append(b)
        elif (
            action_name == "FIRE_CENTER"
            and len(self._bullets) < 4
        ):
            b = self._add_entity(
                "bullet", "!", self._TURRET_CENTER,
                self._GROUND_Y - 2, dx=0, dy=-1,
            )
            self._bullets.append(b)
        elif (
            action_name == "FIRE_RIGHT"
            and len(self._bullets) < 4
        ):
            b = self._add_entity(
                "bullet", "\\", self._TURRET_RIGHT,
                self._GROUND_Y - 2, dx=-1, dy=-1,
            )
            self._bullets.append(b)

        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.x += b.dx
            b.y += b.dy
            if (
                b.y <= 0
                or b.x <= 0
                or b.x >= self._WIDTH - 1
            ):
                b.alive = False

        # Bullet-flyer collisions
        for b in self._bullets:
            if not b.alive:
                continue
            for f in self._flyers:
                if (
                    f.alive
                    and abs(f.x - b.x) <= 1
                    and f.y == b.y
                ):
                    f.alive = False
                    b.alive = False
                    self._on_point_scored(20)
                    reward += 20
                    self._message = "Enemy destroyed! +20"
                    break
        self._bullets = [b for b in self._bullets if b.alive]

        # Spawn flyers
        if self._step_counter % self._spawn_interval == 0:
            y = int(self.rng.integers(2, self._GROUND_Y - 3))
            if self.rng.random() < 0.5:
                f = self._add_entity(
                    "flyer", "→", 1, y, dx=1,
                )
            else:
                f = self._add_entity(
                    "flyer", "←", self._WIDTH - 2, y, dx=-1,
                )
            f.data["bomb_timer"] = int(
                self.rng.integers(5, 15)
            )
            self._flyers.append(f)

        # Move flyers
        for f in self._flyers:
            if not f.alive:
                continue
            f.x += f.dx
            if f.x <= 0 or f.x >= self._WIDTH - 1:
                f.alive = False
                continue
            # Drop bombs on buildings
            f.data["bomb_timer"] = (
                f.data.get("bomb_timer", 10) - 1
            )
            if f.data["bomb_timer"] <= 0:
                f.data["bomb_timer"] = int(
                    self.rng.integers(8, 20)
                )
                # Destroy a building below
                for pos in self._buildings:
                    if pos[0] == f.x:
                        self._buildings.remove(pos)
                        self._message = "Building hit!"
                        break

        self._flyers = [f for f in self._flyers if f.alive]

        # Check game over: all buildings destroyed
        if not self._buildings:
            self._on_life_lost()
            if not self._game_over:
                self._buildings = []
                self._generate_level(self._level)
                self._message = "City damaged! Rebuilding..."

        # Level up every 200 points
        if self._score >= self._level * 200:
            self._level += 1

        self._redraw()
        info["buildings"] = len(self._buildings)
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._GROUND_Y):
            for x in range(self._WIDTH):
                self._set_cell(x, y, " ")
        # Buildings
        for bx, by in self._buildings:
            self._set_cell(bx, by, "H")
        # Turrets
        self._set_cell(self._TURRET_LEFT, self._GROUND_Y - 1, "T")
        self._set_cell(
            self._TURRET_CENTER, self._GROUND_Y - 1, "T"
        )
        self._set_cell(
            self._TURRET_RIGHT, self._GROUND_Y - 1, "T"
        )
        # Flyers
        for f in self._flyers:
            if f.alive:
                self._set_cell(f.x, f.y, f.char)
        # Bullets
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, b.char)

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "border", "=": "ground",
            "T": "turret", "H": "building",
            "→": "enemy flying right",
            "←": "enemy flying left",
            "!": "bullet (vertical)",
            "/": "bullet (diagonal right)",
            "\\": "bullet (diagonal left)",
            " ": "sky",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        bldgs = len(self._buildings)
        enemies = sum(
            1 for f in self._flyers if f.alive
        )
        extra = (
            f"Buildings: {bldgs}  Enemies: {enemies}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Defend Atlantis using three turrets. "
            "FIRE_LEFT shoots diagonally right, "
            "FIRE_CENTER shoots straight up, "
            "FIRE_RIGHT shoots diagonally left. "
            "Protect your buildings from enemy bombers."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Atlantis.\n\n"
            "TASK\n"
            "Defend the city of Atlantis from waves of flying enemies by "
            "firing from three fixed turrets.\n\n"
            "BOARD\n"
            "30 columns by 16 rows. Sky fills rows 1-12; ground '=' is at "
            "row 13. Three turrets 'T' are fixed at columns 3 (left), 15 "
            "(center), 27 (right), sitting on row 12. Six city buildings "
            "'H' stand at columns 7, 10, 13, 17, 20, 23 each 2 cells tall. "
            "Flyers travel horizontally and are rendered as '->' or '<-'. "
            "Your bullets are '/' (left turret, diagonal up-right), '!' "
            "(center turret, straight up), '\\' (right turret, diagonal "
            "up-left).\n\n"
            "MECHANICS\n"
            "You cannot move. Each step choose NOOP or a FIRE action; each "
            "turret can have at most a total of 4 bullets alive. Bullets "
            "move each step in their diagonal or vertical direction; they "
            "die on edges. Flyers spawn every 5-12 steps at either edge in "
            "rows 2 to GROUND-3, flying toward the other side; they drop a "
            "bomb every 5-20 steps that destroys the building directly "
            "below. A bullet within 1 column of a flyer in the same row "
            "kills it.\n\n"
            "SCORING\n"
            "+20 reward per enemy flyer destroyed. No reward for bombs "
            "destroying buildings. Every 200 score points increases the "
            "level and spawn rate.\n\n"
            "TERMINATION\n"
            "When every building is destroyed you lose a life and the city "
            "rebuilds. Episode ends when lives reach 0 or after max_turns.\n\n"
            "HUD\n"
            "Shows score, lives, level, remaining buildings, and active "
            "enemy count.\n\n"
            + self.action_spec.render_for_prompt()
        )
