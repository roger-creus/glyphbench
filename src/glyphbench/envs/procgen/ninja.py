"""Procgen Ninja environment.

Platformer with throwing stars. Agent traverses platforms, defeats
enemies by throwing projectiles, breaks through walls, and reaches goal.

Gym ID: glyphbench/procgen-ninja-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import ProcgenBase


class NinjaEnv(ProcgenBase):
    """Ninja platformer: throw stars at enemies, break walls, reach goal.

    World: 40 wide x 12 tall.  View: 20 x 12.
    Gravity enabled.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "JUMP", "JUMP_RIGHT", "THROW"),
        descriptions=(
            "do nothing this step",
            "move one cell left",
            "move one cell right",
            "jump straight up (if on ground)",
            "jump and move right simultaneously",
            "throw a shuriken in the direction you face",
        ),
    )
    noop_action_name = "NOOP"

    # Reward shaping (Pattern B): +0.4 split across enemy kills, +0.6 on
    # reaching the goal. Cumulative best-case: kill all enemies + reach
    # goal = 1.0. Death gives 0 (Pattern A: no failure penalty).
    _ENEMY_BUDGET = 0.4
    _GOAL_REWARD = 0.6

    def __init__(self, max_turns: int = 512) -> None:
        super().__init__(max_turns=max_turns)
        self._has_gravity = True
        self._view_w = 20
        self._view_h = 12
        self._facing: int = 1  # +1 right, -1 left
        self._enemies_killed: int = 0
        self._total_enemies: int = 0

    def env_id(self) -> str:
        return "glyphbench/procgen-ninja-v0"

    # ------------------------------------------------------------------
    def _generate_level(self, seed: int) -> None:
        W, H = 40, 12
        self._init_world(W, H, fill="\u00b7")
        self._enemies_killed = 0
        self._total_enemies = 0
        ground_y = H - 2

        # Ground
        for x in range(W):
            self._set_cell(x, ground_y, "\u25ac")
            self._set_cell(x, H - 1, "\u25ac")

        # Platforms
        num_plats = int(self.rng.integers(3, 7))
        for _ in range(num_plats):
            px = int(self.rng.integers(4, W - 6))
            py = int(self.rng.integers(ground_y - 5, ground_y - 2))
            pw = int(self.rng.integers(3, 6))
            if py < 1:
                py = 1
            for dx in range(pw):
                if px + dx < W:
                    self._set_cell(px + dx, py, "\u2588")

        # Breakable walls
        num_walls = int(self.rng.integers(1, 4))
        for _ in range(num_walls):
            wx = int(self.rng.integers(8, W - 8))
            if self._world_at(wx, ground_y) == "\u25ac":
                for wy in range(ground_y - 2, ground_y):
                    self._set_cell(wx, wy, "B")

        # Enemies on ground
        num_enemies = int(self.rng.integers(2, 5))
        for _ in range(num_enemies):
            ex = int(self.rng.integers(6, W - 6))
            if self._world_at(ex, ground_y - 1) == "\u00b7":
                self._add_entity("enemy", "E", ex, ground_y - 1, dx=1)
                self._total_enemies += 1

        # Goal
        self._set_cell(W - 2, ground_y - 1, "G")

        # Agent
        self._agent_x = 2
        self._agent_y = ground_y - 1
        self._facing = 1

    # ------------------------------------------------------------------
    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0

        if action_name == "LEFT":
            self._try_move(-1, 0)
            self._facing = -1
            self._agent_dir = (-1, 0)
        elif action_name == "RIGHT":
            self._try_move(1, 0)
            self._facing = 1
            self._agent_dir = (1, 0)
        elif action_name == "JUMP":
            self._start_jump()
            self._agent_dir = (0, -1)
        elif action_name == "JUMP_RIGHT":
            self._start_jump()
            self._try_move(1, 0)
            self._facing = 1
            self._agent_dir = (1, 0)
        elif action_name == "THROW":
            self._add_entity(
                "shuriken", "-", self._agent_x + self._facing, self._agent_y,
                dx=self._facing,
            )

        self._process_jump()

        # Check goal
        ch = self._world_at(self._agent_x, self._agent_y)
        if ch == "G":
            self._message = "Reached the goal!"
            return self._GOAL_REWARD + reward, True, {}

        # Enemy collision with agent
        for e in self._entities:
            if e.alive and e.etype == "enemy" and e.x == self._agent_x and e.y == self._agent_y:
                self._message = "Hit by an enemy!"
                return 0.0, True, {"killed_by": "enemy"}

        return reward, False, {}

    # ------------------------------------------------------------------
    def _advance_entities(self) -> float:
        """Move shurikens and enemies; handle collisions."""
        kills_before = self._enemies_killed
        for e in self._entities:
            if not e.alive:
                continue
            if e.etype == "shuriken":
                e.x += e.dx
                # Out of bounds
                if e.x < 0 or e.x >= self._world_w:
                    e.alive = False
                    continue
                # Hit breakable wall
                ch = self._world_at(e.x, e.y)
                if ch == "B":
                    self._set_cell(e.x, e.y, "\u00b7")
                    e.alive = False
                    continue
                # Hit solid wall
                if self._is_solid(e.x, e.y):
                    e.alive = False
                    continue
                # Hit enemy
                for other in self._entities:
                    if other.alive and other.etype == "enemy" and other.x == e.x and other.y == e.y:
                        other.alive = False
                        e.alive = False
                        self._enemies_killed += 1
                        self._message = "Defeated an enemy!"
            elif e.etype == "enemy":
                nx = e.x + e.dx
                below = self._world_at(nx, e.y + 1)
                if (
                    self._is_solid(nx, e.y)
                    or below not in ("\u25ac", "\u2588")
                    or nx <= 0
                    or nx >= self._world_w - 1
                ):
                    e.dx = -e.dx
                else:
                    e.x = nx

        self._entities = [e for e in self._entities if e.alive]
        kills_this_step = self._enemies_killed - kills_before
        if kills_this_step <= 0 or self._total_enemies <= 0:
            return 0.0
        return kills_this_step * (self._ENEMY_BUDGET / self._total_enemies)

    # ------------------------------------------------------------------
    def _is_solid(self, x: int, y: int) -> bool:
        ch = self._world_at(x, y)
        return ch in ("\u2588", "\u25ac", "+", "|", "-", "B")

    def _symbol_meaning(self, ch: str) -> str:
        m: dict[str, str] = {
            "\u00b7": "empty",
            "\u25ac": "ground",
            "\u2588": "platform",
            "B": "breakable wall",
            "G": "goal (finishes the level)",
            "@": "you",
        }
        return m.get(ch, ch)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        facing = "right" if self._facing == 1 else "left"
        enemies = sum(
            1 for e in self._entities
            if e.alive and e.etype == "enemy"
        )
        extra = (
            f"Facing: {facing}"
            f"  Kills: {self._enemies_killed}"
            f"  Enemies: {enemies}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Traverse the level as a ninja. Throw shurikens (THROW) to "
            "defeat enemies (E) and break walls (B). Killing every enemy "
            "yields +0.4 total; reaching the goal (G) yields +0.6 (best "
            "case +1.0). Avoid touching enemies."
        )
