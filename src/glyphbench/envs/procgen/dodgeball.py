"""Procgen Dodgeball environment.

Arena where the agent throws balls at enemies while dodging.

Gym ID: glyphbench/procgen-dodgeball-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import ProcgenBase


class DodgeballEnv(ProcgenBase):
    """Procgen Dodgeball: throw balls to hit wandering enemies."""

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN", "THROW"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "move up",
            "move down",
            "throw ball in facing direction",
        ),
    )

    GRID_W = 14
    GRID_H = 12
    # Reward shaping (Pattern D): first _WIN_TARGET kills each yield
    # +1/_WIN_TARGET; subsequent kills add 0 (cumulative caps at +1.0).
    # Wave-clear bonuses removed; terminal -1.0 on getting hit.
    _WIN_TARGET = 20
    _DEATH_PENALTY = -1.0

    def env_id(self) -> str:
        return "glyphbench/procgen-dodgeball-v0"

    def _generate_level(self, seed: int) -> None:
        w, h = self.GRID_W, self.GRID_H
        self._init_world(w, h, fill=" ")

        # Walls around the arena
        for x in range(w):
            self._set_cell(x, 0, "\u2588")
            self._set_cell(x, h - 1, "\u2588")
        for y in range(h):
            self._set_cell(0, y, "\u2588")
            self._set_cell(w - 1, y, "\u2588")

        # Agent starts center-bottom
        self._agent_x = w // 2
        self._agent_y = h - 3
        self._facing_dx = 0
        self._facing_dy = -1  # facing up by default

        self._enemies_killed = 0
        self._level = 1

        # Spawn enemies
        self._spawn_enemies(count=4)

    def _spawn_enemies(self, count: int) -> None:
        """Spawn enemies at random positions away from agent."""
        w, h = self.GRID_W, self.GRID_H
        for _ in range(count):
            for _attempt in range(50):
                x = int(self.rng.integers(2, w - 2))
                y = int(self.rng.integers(2, h - 2))
                dist = abs(x - self._agent_x) + abs(y - self._agent_y)
                if dist > 3:
                    ddx = 1 if int(self.rng.integers(0, 2)) == 0 else -1
                    ddy = 0
                    self._add_entity("enemy", "E", x, y, dx=ddx, dy=ddy)
                    break

    def _advance_entities(self) -> float:
        """Move entities. Enemies wander randomly, balls fly straight."""
        for e in self._entities:
            if not e.alive:
                continue
            if e.etype == "ball":
                e.x += e.dx
                e.y += e.dy
                # Remove if hitting a wall
                if self._is_solid(e.x, e.y):
                    e.alive = False
            elif e.etype == "enemy":
                # Random wander: change direction occasionally
                if float(self.rng.random()) < 0.3:
                    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    idx = int(self.rng.integers(0, len(dirs)))
                    e.dx, e.dy = dirs[idx]
                nx, ny = e.x + e.dx, e.y + e.dy
                if not self._is_solid(nx, ny):
                    e.x = nx
                    e.y = ny
                else:
                    e.dx = -e.dx
                    e.dy = -e.dy
        self._entities = [e for e in self._entities if e.alive]
        return 0.0

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        # Movement
        if action_name == "LEFT":
            self._try_move(-1, 0)
            self._facing_dx, self._facing_dy = -1, 0
        elif action_name == "RIGHT":
            self._try_move(1, 0)
            self._facing_dx, self._facing_dy = 1, 0
        elif action_name == "UP":
            self._try_move(0, -1)
            self._facing_dx, self._facing_dy = 0, -1
        elif action_name == "DOWN":
            self._try_move(0, 1)
            self._facing_dx, self._facing_dy = 0, 1
        elif action_name == "THROW":
            bx = self._agent_x + self._facing_dx
            by = self._agent_y + self._facing_dy
            if not self._is_solid(bx, by):
                self._add_entity(
                    "ball", "*", bx, by,
                    dx=self._facing_dx, dy=self._facing_dy,
                )

        # Check ball-enemy collisions
        for ball in [e for e in self._entities if e.etype == "ball" and e.alive]:
            for enemy in [
                e for e in self._entities if e.etype == "enemy" and e.alive
            ]:
                if ball.x == enemy.x and ball.y == enemy.y:
                    ball.alive = False
                    enemy.alive = False
                    if self._enemies_killed < self._WIN_TARGET:
                        reward += 1.0 / self._WIN_TARGET
                    self._enemies_killed += 1

        # Check agent-enemy collision (terminal failure -> -1.0).
        for e in self._entities:
            if not e.alive or e.etype != "enemy":
                continue
            if e.x == self._agent_x and e.y == self._agent_y:
                reward = self._DEATH_PENALTY
                terminated = True
                self._message = "Hit by an enemy!"
                return reward, terminated, info

        # Clean dead entities
        self._entities = [e for e in self._entities if e.alive]

        # When a wave is cleared, spawn the next one (no extra reward —
        # progress is already paid per kill).
        enemies_alive = sum(1 for e in self._entities if e.etype == "enemy")
        if enemies_alive == 0:
            self._level += 1
            self._message = f"Level {self._level - 1} cleared!"
            self._spawn_enemies(count=3 + self._level)

        info["enemies_killed"] = self._enemies_killed
        info["level"] = self._level
        return reward, terminated, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        dirs = {
            (1, 0): "right", (-1, 0): "left",
            (0, -1): "up", (0, 1): "down",
        }
        facing = dirs.get(
            (self._facing_dx, self._facing_dy), "up"
        )
        enemies = sum(
            1 for e in self._entities
            if e.alive and e.etype == "enemy"
        )
        extra = (
            f"Facing: {facing}"
            f"  Enemies: {enemies}"
            f"  Wave: {self._level}"
            f"  Kills: {self._enemies_killed}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "You are in an arena (@). Throw balls (*) at enemies (E) by moving "
            "in a direction then using THROW. The first "
            f"{self._WIN_TARGET} enemy kills each yield +1/{self._WIN_TARGET}; "
            "extra kills add nothing. Touching an enemy ends the episode at -1."
        )

    def _symbol_meaning(self, ch: str) -> str:
        meanings = {
            " ": "arena floor",
            "\u2588": "wall",
            "E": "enemy",
            "*": "ball",
        }
        return meanings.get(ch, super()._symbol_meaning(ch))
