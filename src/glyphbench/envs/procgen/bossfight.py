"""Procgen BossFight environment.

Space arena with a multi-phase boss. Agent at bottom dodges boss projectiles
and fires back.

Gym ID: glyphbench/procgen-bossfight-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import ProcgenBase

_BOSS_MAX_HP = 10


class BossFightEnv(ProcgenBase):
    """Procgen BossFight: multi-phase boss with projectile patterns."""

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "move up",
            "move down",
            "fire bullet upward",
        ),
    )

    GRID_W = 20
    GRID_H = 14
    # Reward shaping (Pattern D): +0.5 distributed across the _BOSS_MAX_HP
    # hits + +0.5 on defeating the boss = +1.0 best case; terminal -1.0
    # on getting hit by a boss projectile.
    _HIT_BUDGET = 0.5
    _DEFEAT_REWARD = 0.5
    _DEATH_PENALTY = -1.0

    def env_id(self) -> str:
        return "glyphbench/procgen-bossfight-v0"

    def _generate_level(self, seed: int) -> None:
        w, h = self.GRID_W, self.GRID_H
        self._init_world(w, h, fill=" ")

        # Agent at bottom center
        self._agent_x = w // 2
        self._agent_y = h - 2

        # Boss at top center
        self._boss_x = w // 2
        self._boss_y = 1
        self._boss_hp = _BOSS_MAX_HP
        self._boss_dir = 1  # horizontal movement direction
        self._boss_timer = 0
        self._hits_on_boss = 0

    def _boss_phase(self) -> int:
        """Determine boss phase based on HP."""
        if self._boss_hp > 7:
            return 1
        if self._boss_hp > 3:
            return 2
        return 3

    def _boss_attack(self) -> None:
        """Boss fires projectiles based on current phase."""
        phase = self._boss_phase()
        self._boss_timer += 1

        if phase == 1:
            # Phase 1: single shot downward every 4 steps
            if self._boss_timer % 4 == 0:
                self._add_entity(
                    "boss_bullet", "v", self._boss_x, self._boss_y + 1,
                    dx=0, dy=1,
                )
        elif phase == 2:
            # Phase 2: spread shot every 3 steps
            if self._boss_timer % 3 == 0:
                self._add_entity(
                    "boss_bullet", "v", self._boss_x, self._boss_y + 1,
                    dx=0, dy=1,
                )
                self._add_entity(
                    "boss_bullet", "\\", self._boss_x + 1, self._boss_y + 1,
                    dx=1, dy=1,
                )
                self._add_entity(
                    "boss_bullet", "/", self._boss_x - 1, self._boss_y + 1,
                    dx=-1, dy=1,
                )
        else:
            # Phase 3: rapid fire every 2 steps + aimed shots
            if self._boss_timer % 2 == 0:
                self._add_entity(
                    "boss_bullet", "v", self._boss_x, self._boss_y + 1,
                    dx=0, dy=1,
                )
                # Aimed shot toward agent
                aim_dx = 0
                if self._agent_x < self._boss_x:
                    aim_dx = -1
                elif self._agent_x > self._boss_x:
                    aim_dx = 1
                self._add_entity(
                    "boss_bullet", "o", self._boss_x + aim_dx,
                    self._boss_y + 1, dx=aim_dx, dy=1,
                )

    def _advance_entities(self) -> float:
        """Move all entities. Boss moves horizontally."""
        # Move boss
        self._boss_x += self._boss_dir
        if self._boss_x <= 1 or self._boss_x >= self.GRID_W - 2:
            self._boss_dir = -self._boss_dir

        # Move entities
        for e in self._entities:
            if not e.alive:
                continue
            e.x += e.dx
            e.y += e.dy
            if e.x < 0 or e.x >= self.GRID_W or e.y < 0 or e.y >= self.GRID_H:
                e.alive = False
        self._entities = [e for e in self._entities if e.alive]
        return 0.0

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        # Movement (constrain to arena)
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
            by = self._agent_y - 1
            if by >= 0:
                self._add_entity(
                    "bullet", "|", self._agent_x, by, dx=0, dy=-1,
                )

        # Boss attacks
        self._boss_attack()

        # Check player bullet hits on boss (each hit pays +0.5/_BOSS_MAX_HP).
        for bullet in [e for e in self._entities if e.etype == "bullet" and e.alive]:
            if bullet.x == self._boss_x and bullet.y == self._boss_y:
                bullet.alive = False
                self._boss_hp -= 1
                self._hits_on_boss += 1
                reward += self._HIT_BUDGET / _BOSS_MAX_HP
                self._message = f"Hit! Boss HP: {self._boss_hp}/{_BOSS_MAX_HP}"

        # Check boss death
        if self._boss_hp <= 0:
            reward += self._DEFEAT_REWARD
            terminated = True
            self._message = "Boss defeated!"
            return reward, terminated, info

        # Check boss bullet hits on agent (terminal failure -> -1.0).
        for e in self._entities:
            if not e.alive or e.etype != "boss_bullet":
                continue
            if e.x == self._agent_x and e.y == self._agent_y:
                reward = self._DEATH_PENALTY
                terminated = True
                self._message = "Hit by boss projectile!"
                return reward, terminated, info

        # Clean dead entities
        self._entities = [e for e in self._entities if e.alive]

        info["boss_hp"] = self._boss_hp
        info["boss_phase"] = self._boss_phase()
        info["hits_on_boss"] = self._hits_on_boss
        return reward, terminated, info

    def _render_current_observation(self) -> GridObservation:
        """Override to render boss and add boss state to HUD."""
        if self._boss_hp > 0:
            boss_entity = self._add_entity(
                "boss_render", "B", self._boss_x, self._boss_y,
            )
            obs = super()._render_current_observation()
            boss_entity.alive = False
            self._entities = [
                e for e in self._entities if e.alive
            ]
        else:
            obs = super()._render_current_observation()
        phase = self._boss_phase()
        # Boss x/y removed — boss glyph is visible on the grid.
        extra = (
            f"Boss: HP={self._boss_hp}/{_BOSS_MAX_HP}"
            f"  phase={phase}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "You face a boss (B) in a space arena. FIRE shoots bullets (|) "
            "upward. The boss has 3 phases as HP decreases: single shots, "
            "spread shots, rapid aimed fire. Dodge boss projectiles (v, \\, /, o). "
            f"Each hit yields +0.5/{_BOSS_MAX_HP}; defeating the boss yields "
            f"+0.5 (best case +1.0). Getting hit ends the episode at -1.0. "
            f"Boss HP: {_BOSS_MAX_HP}."
        )

    def _symbol_meaning(self, ch: str) -> str:
        meanings = {
            " ": "space",
            "B": "boss",
            "|": "your bullet",
            "v": "boss bullet (down)",
            "\\": "boss bullet (diagonal)",
            "/": "boss bullet (diagonal)",
            "o": "boss aimed shot",
        }
        return meanings.get(ch, super()._symbol_meaning(ch))
