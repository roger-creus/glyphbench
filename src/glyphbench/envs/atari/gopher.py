"""Atari Gopher environment.

Defend carrots from gophers digging underground.

Gym ID: glyphbench/atari-gopher-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity

class GopherEnv(AtariBase):
    """Gopher: protect your carrot garden from gophers.

    20x16 grid. Carrots on the surface. Gophers dig tunnels
    from below to eat them. Fill holes to block gophers.

    Actions: NOOP, LEFT, RIGHT, FILL
    Pattern D: +1/_WIN_TARGET per gopher bonked/trapped, -1 if
    all carrots eaten (full-scope = 8 gophers neutralized).
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FILL"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "fill the hole below you",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 16
    _SURFACE_Y = 10
    _PLAYER_Y = 9

    # Pattern D full-scope target: 8 gophers neutralized.
    _WIN_TARGET: int = 8
    _DEATH_PENALTY: float = -1.0

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._gophers: list[AtariEntity] = []
        self._carrots: list[tuple[int, int]] = []
        self._holes: set[tuple[int, int]] = set()
        self._step_counter: int = 0
        self._spawn_interval: int = 15
        self._progress_count: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-gopher-v0"

    def _reset(self, seed: int):
        self._progress_count = 0
        return super()._reset(seed)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._gophers = []
        self._holes = set()
        self._step_counter = 0
        self._spawn_interval = max(8, 18 - self._level * 2)

        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")

        # Ground layer
        for x in range(1, self._WIDTH - 1):
            self._set_cell(x, self._SURFACE_Y, "=")
            for y in range(self._SURFACE_Y + 1, self._HEIGHT - 1):
                self._set_cell(x, y, "·")

        # Carrots on the surface
        self._carrots = []
        for x in range(3, self._WIDTH - 3, 2):
            pos = (x, self._SURFACE_Y - 1)
            self._carrots.append(pos)

        self._player_x = self._WIDTH // 2
        self._player_y = self._PLAYER_Y

        self._redraw()

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        if action_name == "LEFT" and self._player_x > 1:
            self._player_x -= 1
            self._player_dir = (-1, 0)
        elif (
            action_name == "RIGHT"
            and self._player_x < self._WIDTH - 2
        ):
            self._player_x += 1
            self._player_dir = (1, 0)
        elif action_name == "FILL":
            # Fill hole below player
            hole_pos = (self._player_x, self._SURFACE_Y)
            if hole_pos in self._holes:
                self._holes.discard(hole_pos)
                self._message = "Hole filled!"
            # Bonk gopher if at player position
            for g in self._gophers:
                if (
                    g.alive
                    and g.x == self._player_x
                    and g.y <= self._SURFACE_Y
                ):
                    g.alive = False
                    self._on_point_scored(1)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._message = "Gopher bonked!"

        # Spawn gophers
        if self._step_counter % self._spawn_interval == 0:
            gx = int(self.rng.integers(2, self._WIDTH - 2))
            g = self._add_entity(
                "gopher", "G", gx, self._HEIGHT - 2,
            )
            g.data["target_x"] = int(
                self.rng.integers(2, self._WIDTH - 2)
            )
            g.data["state"] = "digging_up"
            self._gophers.append(g)

        # Move gophers
        if self._step_counter % 2 == 0:
            for g in self._gophers:
                if not g.alive:
                    continue
                state = g.data.get("state", "digging_up")
                if state == "digging_up":
                    # Move toward surface
                    tx = g.data.get("target_x", g.x)
                    if g.x < tx:
                        g.x += 1
                    elif g.x > tx:
                        g.x -= 1
                    elif g.y > self._SURFACE_Y:
                        g.y -= 1
                    else:
                        # Create hole at surface
                        self._holes.add(
                            (g.x, self._SURFACE_Y)
                        )
                        g.data["state"] = "eating"
                        g.data["eat_timer"] = 4
                elif state == "eating":
                    # Try to eat a carrot
                    g.data["eat_timer"] = (
                        g.data.get("eat_timer", 4) - 1
                    )
                    if g.data["eat_timer"] <= 0:
                        # Eat carrot above
                        carrot_pos = (g.x, self._SURFACE_Y - 1)
                        if carrot_pos in self._carrots:
                            self._carrots.remove(carrot_pos)
                            self._message = "Carrot eaten!"
                        g.data["state"] = "retreating"
                elif state == "retreating":
                    g.y += 1
                    if g.y >= self._HEIGHT - 1:
                        g.alive = False

                # Check if hole is filled on gopher
                if (
                    g.alive
                    and g.y == self._SURFACE_Y
                    and (g.x, self._SURFACE_Y) not in self._holes
                ):
                    g.alive = False
                    self._on_point_scored(1)
                    if self._progress_count < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                        self._progress_count += 1
                    self._message = "Gopher trapped!"

        self._gophers = [g for g in self._gophers if g.alive]

        # Game over if all carrots gone (Pattern D death penalty)
        if not self._carrots and not self._game_over:
            self._on_life_lost()
            reward = self._DEATH_PENALTY
            self._message = "Garden lost!"

        # Win check
        if self._progress_count >= self._WIN_TARGET and not self._game_over:
            self._game_over = True
            info["won"] = True
            self._message = "Garden saved!"

        # Level up every 10 points
        if self._score > 0 and self._score >= self._level * 10:
            self._level += 1

        self._redraw()
        info["carrots"] = len(self._carrots)
        return reward, self._game_over, info

    def _redraw(self) -> None:
        # Clear above ground
        for y in range(1, self._SURFACE_Y):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")
        # Underground
        for y in range(self._SURFACE_Y, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, "·")
        # Surface
        for x in range(1, self._WIDTH - 1):
            if (x, self._SURFACE_Y) in self._holes:
                self._set_cell(x, self._SURFACE_Y, "O")
            else:
                self._set_cell(x, self._SURFACE_Y, "=")
        # Carrots
        for cx, cy in self._carrots:
            self._set_cell(cx, cy, "↑")
        # Gophers
        for g in self._gophers:
            if g.alive:
                self._set_cell(g.x, g.y, "G")

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "wall", "│": "wall",
            "=": "ground", "·": "underground",
            "↑": "carrot", "G": "gopher",
            "O": "hole", " ": "sky",
        }.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        extra = (
            f"Carrots: {len(self._carrots)}  "
            f"Holes: {len(self._holes)}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Protect your carrots from gophers. "
            "Move above holes and FILL to block them. "
            "Bonk gophers with FILL when they surface."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari Gopher.\n\n"
            "TASK\n"
            "Defend a row of carrots from burrowing gophers by "
            "filling holes and bonking gophers as they surface.\n\n"
            "BOARD\n"
            "20 columns by 16 rows, walled at edges. Surface '=' at "
            "row 10; sky above, underground '.' below. Carrots are "
            "spaced along the surface in a single row. Gophers are "
            "'G'. Holes in the surface appear as 'O'. You walk just "
            "above the surface as an arrow glyph.\n\n"
            "MECHANICS\n"
            "LEFT / RIGHT move you 1 cell along the row. FILL does "
            "two things: (1) if the cell directly below you is a "
            "hole, fill it; (2) bonk any gopher at your column with "
            "y <= surface. Gophers spawn at the bottom every "
            "max(8, 18 - 2*level) steps, dig toward a random target "
            "x, create a hole when they reach the surface, then eat "
            "a carrot after a 4-step timer, then retreat. If a "
            "gopher's hole is filled while it is on the surface it "
            "dies.\n\n"
            "SCORING\n"
            "+1/8 reward per gopher bonked or trapped (Pattern D "
            "full-scope = 8 gophers). -1.0 if all carrots are "
            "eaten (terminates).\n\n"
            "TERMINATION\n"
            "Episode ends when all carrots are eaten (failure, -1) "
            "or after 8 gophers neutralized (cumulative reward "
            "plateaus at +1.0) or after max_turns.\n\n"
            "HUD\n"
            "Shows score, lives, level, carrots remaining, and "
            "number of holes.\n\n"
            + self.action_spec.render_for_prompt()
        )
