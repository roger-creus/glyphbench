"""Procgen StarPilot environment.

Horizontal shoot-em-up. Agent on left, enemies approach from right.

Gym ID: glyphbench/procgen-starpilot-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import ProcgenBase


class StarPilotEnv(ProcgenBase):
    """Procgen StarPilot: horizontal shooter with power-ups."""

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "move up",
            "move down",
            "fire bullet right",
        ),
    )

    GRID_W = 40
    GRID_H = 12
    # Reward shaping (Pattern D): cumulative best case +1.0 (kill the first
    # _ENEMY_TARGET enemies AND grab _POWERUP_TARGET power-ups), worst case
    # -1.0 (terminal collision with an enemy ship).
    _ENEMY_TARGET = 20
    _ENEMY_BUDGET = 0.8
    _POWERUP_TARGET = 5
    _POWERUP_BUDGET = 0.2
    _DEATH_PENALTY = -1.0

    def env_id(self) -> str:
        return "glyphbench/procgen-starpilot-v0"

    def _generate_level(self, seed: int) -> None:
        w, h = self.GRID_W, self.GRID_H
        self._init_world(w, h, fill=" ")

        # Agent on the left side
        self._agent_x = 2
        self._agent_y = h // 2

        self._enemies_killed = 0
        self._powerups_collected = 0
        self._spawn_timer = 0
        self._scroll_offset = 0

    def _maybe_spawn(self) -> None:
        """Spawn enemies and power-ups from the right edge."""
        w, h = self.GRID_W, self.GRID_H
        self._spawn_timer += 1

        # Spawn enemy every 3-5 steps
        if self._spawn_timer % 3 == 0:
            ey = int(self.rng.integers(1, h - 1))
            speed = -1 if int(self.rng.integers(0, 3)) > 0 else -2
            etype = "enemy"
            char = "E"
            # Occasionally spawn a fast enemy
            if float(self.rng.random()) < 0.2:
                char = "V"
                speed = -2
            self._add_entity(etype, char, w - 1, ey, dx=speed, dy=0)

        # Spawn power-up occasionally
        if self._spawn_timer % 12 == 0:
            py = int(self.rng.integers(1, h - 1))
            self._add_entity("powerup", "$", w - 1, py, dx=-1, dy=0)

    def _advance_entities(self) -> float:
        """Move all entities. Remove out-of-bounds ones."""
        for e in self._entities:
            if not e.alive:
                continue
            e.x += e.dx
            e.y += e.dy
            # Remove if out of bounds
            if e.x < 0 or e.x >= self.GRID_W or e.y < 0 or e.y >= self.GRID_H:
                e.alive = False

            # Enemies: slight vertical wobble
            if e.etype == "enemy" and e.alive and float(self.rng.random()) < 0.15:
                wobble = 1 if int(self.rng.integers(0, 2)) == 0 else -1
                ny = e.y + wobble
                if 0 <= ny < self.GRID_H:
                    e.y = ny

        self._entities = [e for e in self._entities if e.alive]
        return 0.0

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        # Movement -- constrain to left portion
        if action_name == "LEFT":
            self._agent_dir = (-1, 0)
            nx = self._agent_x - 1
            if nx >= 0:
                self._agent_x = nx
        elif action_name == "RIGHT":
            self._agent_dir = (1, 0)
            nx = self._agent_x + 1
            if nx < self.GRID_W:
                self._agent_x = nx
        elif action_name == "UP":
            self._agent_dir = (0, -1)
            ny = self._agent_y - 1
            if ny >= 0:
                self._agent_y = ny
        elif action_name == "DOWN":
            self._agent_dir = (0, 1)
            ny = self._agent_y + 1
            if ny < self.GRID_H:
                self._agent_y = ny
        elif action_name == "FIRE":
            bx = self._agent_x + 1
            if bx < self.GRID_W:
                self._add_entity(
                    "bullet", "-", bx, self._agent_y, dx=2, dy=0,
                )

        # Spawn new entities
        self._maybe_spawn()

        # Check bullet-enemy collisions
        for bullet in [e for e in self._entities if e.etype == "bullet" and e.alive]:
            for enemy in [
                e for e in self._entities if e.etype == "enemy" and e.alive
            ]:
                if bullet.x == enemy.x and bullet.y == enemy.y:
                    bullet.alive = False
                    enemy.alive = False
                    if self._enemies_killed < self._ENEMY_TARGET:
                        reward += self._ENEMY_BUDGET / self._ENEMY_TARGET
                    self._enemies_killed += 1

        # Check agent picks up power-up
        for e in self._entities:
            if not e.alive or e.etype != "powerup":
                continue
            if e.x == self._agent_x and e.y == self._agent_y:
                e.alive = False
                if self._powerups_collected < self._POWERUP_TARGET:
                    reward += self._POWERUP_BUDGET / self._POWERUP_TARGET
                self._powerups_collected += 1
                self._message = "Power-up collected!"

        # Check agent-enemy collision (terminal failure -> -1.0).
        for e in self._entities:
            if not e.alive or e.etype != "enemy":
                continue
            if e.x == self._agent_x and e.y == self._agent_y:
                reward = self._DEATH_PENALTY
                terminated = True
                self._message = "Destroyed by an enemy!"
                return reward, terminated, info

        # Clean dead entities
        self._entities = [e for e in self._entities if e.alive]

        info["enemies_killed"] = self._enemies_killed
        info["powerups_collected"] = self._powerups_collected
        return reward, terminated, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        alive = sum(
            1 for e in self._entities
            if e.alive and e.etype == "enemy"
        )
        extra = (
            f"Enemies: {alive}"
            f"  Kills: {self._enemies_killed}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "You pilot a ship (@) in a horizontal shoot-em-up. "
            "Enemies (E, V) approach across the field. FIRE shoots bullets (-). "
            f"Each of the first {self._ENEMY_TARGET} kills yields "
            f"+0.8/{self._ENEMY_TARGET}; each of the first "
            f"{self._POWERUP_TARGET} power-ups ($) yields "
            f"+0.2/{self._POWERUP_TARGET} (best case +1.0). Colliding with "
            "an enemy ends the episode at -1."
        )

    def _symbol_meaning(self, ch: str) -> str:
        meanings = {
            " ": "space",
            "E": "enemy ship",
            "V": "fast enemy",
            "-": "bullet",
            "$": "power-up (small reward)",
        }
        return meanings.get(ch, super()._symbol_meaning(ch))
