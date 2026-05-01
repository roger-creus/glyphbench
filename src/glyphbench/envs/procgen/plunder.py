"""Procgen Plunder environment.

Naval top-down shooter. Agent ship at bottom, pirate ships from top.
Shoot pirates, avoid hitting civilian ships.

Gym ID: glyphbench/procgen-plunder-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import ProcgenBase


class PlunderEnv(ProcgenBase):
    """Procgen Plunder: naval shooter, sink pirates, spare civilians."""

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "FIRE"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "move up",
            "fire cannon upward",
        ),
    )

    GRID_W = 20
    GRID_H = 14
    # Reward shaping (Pattern D):
    # Each of the first _WIN_TARGET pirates pays +1/_WIN_TARGET; civilians
    # pay -1/_WIN_TARGET. Progress is clamped to [0, 1] over the episode so
    # bad shooting cannot drive us below 0. Terminal collision pays -1.0.
    _WIN_TARGET = 20
    _DEATH_PENALTY = -1.0

    def env_id(self) -> str:
        return "glyphbench/procgen-plunder-v0"

    def _generate_level(self, seed: int) -> None:
        w, h = self.GRID_W, self.GRID_H
        self._init_world(w, h, fill="\u2248")

        # Agent ship at bottom center
        self._agent_x = w // 2
        self._agent_y = h - 2

        self._pirates_sunk = 0
        self._civilians_hit = 0
        self._spawn_timer = 0
        # Net cumulative progress paid out so far in [0, 1]. Used to clamp
        # the per-step delta so cumulative reward never escapes the budget.
        self._progress_paid = 0.0

    def _maybe_spawn(self) -> None:
        """Spawn pirate and civilian ships from the top."""
        w = self.GRID_W
        self._spawn_timer += 1

        # Spawn pirate every 4 steps
        if self._spawn_timer % 4 == 0:
            px = int(self.rng.integers(1, w - 1))
            self._add_entity("pirate", "P", px, 0, dx=0, dy=1)

        # Spawn civilian every 7 steps
        if self._spawn_timer % 7 == 0:
            cx = int(self.rng.integers(1, w - 1))
            self._add_entity("civilian", "c", cx, 0, dx=0, dy=1)

    def _advance_entities(self) -> float:
        """Move entities: cannonballs up, ships down."""
        for e in self._entities:
            if not e.alive:
                continue
            e.x += e.dx
            e.y += e.dy
            # Remove if out of bounds
            if e.y < 0 or e.y >= self.GRID_H or e.x < 0 or e.x >= self.GRID_W:
                e.alive = False

            # Pirates: slight horizontal drift
            if e.etype == "pirate" and e.alive and float(self.rng.random()) < 0.2:
                drift = 1 if int(self.rng.integers(0, 2)) == 0 else -1
                nx = e.x + drift
                if 0 <= nx < self.GRID_W:
                    e.x = nx

        self._entities = [e for e in self._entities if e.alive]
        return 0.0

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        # Movement (horizontal + forward)
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
        elif action_name == "FIRE":
            by = self._agent_y - 1
            if by >= 0:
                self._add_entity(
                    "cannonball", "^", self._agent_x, by, dx=0, dy=-1,
                )

        # Spawn new ships
        self._maybe_spawn()

        # Check cannonball-ship collisions. Pirate kill adds positive
        # progress (clamped at +1.0); civilian hit subtracts (floored at 0).
        unit = 1.0 / self._WIN_TARGET
        for ball in [
            e for e in self._entities if e.etype == "cannonball" and e.alive
        ]:
            for ship in [
                e
                for e in self._entities
                if e.etype in ("pirate", "civilian") and e.alive
            ]:
                if ball.x == ship.x and ball.y == ship.y:
                    ball.alive = False
                    ship.alive = False
                    if ship.etype == "pirate":
                        new_progress = min(1.0, self._progress_paid + unit)
                        reward += new_progress - self._progress_paid
                        self._progress_paid = new_progress
                        self._pirates_sunk += 1
                    else:
                        new_progress = max(0.0, self._progress_paid - unit)
                        reward += new_progress - self._progress_paid
                        self._progress_paid = new_progress
                        self._civilians_hit += 1
                        self._message = "You hit a civilian ship!"

        # Check agent collision with ships (terminal failure -> -1.0).
        for e in self._entities:
            if not e.alive or e.etype not in ("pirate", "civilian"):
                continue
            if e.x == self._agent_x and e.y == self._agent_y:
                reward = self._DEATH_PENALTY - self._progress_paid
                self._progress_paid = 0.0
                terminated = True
                self._message = "Rammed by a ship!"
                return reward, terminated, info

        # Clean dead entities
        self._entities = [e for e in self._entities if e.alive]

        info["pirates_sunk"] = self._pirates_sunk
        info["civilians_hit"] = self._civilians_hit
        return reward, terminated, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        pirates = sum(
            1 for e in self._entities
            if e.alive and e.etype == "pirate"
        )
        civilians = sum(
            1 for e in self._entities
            if e.alive and e.etype == "civilian"
        )
        extra = (
            f"Pirates sunk: {self._pirates_sunk}"
            f"  Civilians hit: {self._civilians_hit}"
            f"  Nearby: {pirates}P {civilians}c"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "You captain a ship (@) on the sea (\u2248). Pirate ships (P) "
            "approach across the water; FIRE launches cannonballs (^). "
            f"Each of the first {self._WIN_TARGET} pirates yields "
            f"+1/{self._WIN_TARGET}; hitting a civilian (c) reverses one "
            "unit of progress. Colliding with any ship ends the episode "
            "at a net cumulative -1."
        )

    def _symbol_meaning(self, ch: str) -> str:
        meanings = {
            "\u2248": "sea",
            "P": "pirate ship (advances progress)",
            "c": "civilian ship (reverses progress if hit)",
            "^": "cannonball",
        }
        return meanings.get(ch, super()._symbol_meaning(ch))
