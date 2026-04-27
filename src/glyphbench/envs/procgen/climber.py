"""Procgen Climber environment.

Vertical platformer. Agent climbs upward collecting stars on platforms,
avoiding patrolling enemies, and reaching the top for a big reward.

Gym ID: glyphbench/procgen-climber-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import ProcgenBase


class ClimberEnv(ProcgenBase):
    """Vertical platformer: climb platforms, collect stars, reach the top.

    World: 14 wide x 40 tall.  View: 14 x 20.
    Gravity enabled; agent falls when not on solid ground.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "JUMP", "JUMP_LEFT", "JUMP_RIGHT"),
        descriptions=(
            "do nothing this step",
            "move one cell left",
            "move one cell right",
            "jump straight up (if on ground)",
            "jump and move left simultaneously",
            "jump and move right simultaneously",
        ),
    )
    noop_action_name = "NOOP"

    def __init__(self, max_turns: int = 512) -> None:
        super().__init__(max_turns=max_turns)
        self._has_gravity = True
        self._view_w = 14
        self._view_h = 20
        self._stars_collected: int = 0
        self._total_stars: int = 0

    def env_id(self) -> str:
        return "glyphbench/procgen-climber-v0"

    # ------------------------------------------------------------------
    def _generate_level(self, seed: int) -> None:
        W, H = 14, 40
        self._init_world(W, H, fill="\u00b7")
        self._total_stars = 0
        self._stars_collected = 0

        # Bottom ground
        for x in range(W):
            self._set_cell(x, H - 1, "\u25ac")

        # Place platforms rising upward, every 3-5 rows
        y = H - 4
        plat_idx = 0
        while y > 2:
            gap = int(self.rng.integers(2, 4))
            px = int(self.rng.integers(1, W - 5))
            pw = int(self.rng.integers(4, min(8, W - px)))
            for dx in range(pw):
                self._set_cell(px + dx, y, "\u25ac")

            # Star on platform
            star_x = px + int(self.rng.integers(0, pw))
            self._set_cell(star_x, y - 1, "*")
            self._total_stars += 1

            # Enemy on every other platform
            if plat_idx % 2 == 1 and pw >= 4:
                ex = px + 1
                self._add_entity("enemy", "E", ex, y - 1, dx=1)

            y -= gap
            plat_idx += 1

        # Top goal row
        for x in range(W):
            self._set_cell(x, 1, "\u25ac")
        self._set_cell(W // 2, 0, "G")

        # Walls on sides
        for row in range(H):
            self._set_cell(0, row, "\u2588")
            self._set_cell(W - 1, row, "\u2588")

        # Agent starts on bottom ground
        self._agent_x = W // 2
        self._agent_y = H - 2

    # ------------------------------------------------------------------
    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0

        # Movement
        if action_name == "LEFT":
            self._try_move(-1, 0)
        elif action_name == "RIGHT":
            self._try_move(1, 0)
        elif action_name == "JUMP":
            self._start_jump()
            self._agent_dir = (0, -1)
        elif action_name == "JUMP_LEFT":
            self._start_jump()
            self._try_move(-1, 0)
            self._agent_dir = (-1, 0)
        elif action_name == "JUMP_RIGHT":
            self._start_jump()
            self._try_move(1, 0)
            self._agent_dir = (1, 0)

        self._process_jump()

        # Collect star
        ch = self._world_at(self._agent_x, self._agent_y)
        if ch == "*":
            self._set_cell(self._agent_x, self._agent_y, "\u00b7")
            reward += 1.0
            self._stars_collected += 1
            self._message = "Collected a star!"

        # Goal
        if ch == "G":
            reward += 5.0
            self._message = "Reached the top!"
            return reward, True, {}

        # Enemy collision
        for e in self._entities:
            if e.alive and e.x == self._agent_x and e.y == self._agent_y:
                self._message = "Hit by an enemy!"
                return 0.0, True, {"killed_by": "enemy"}

        # Fell off bottom
        if self._agent_y >= self._world_h - 1:
            pass  # on ground, fine

        return reward, False, {}

    # ------------------------------------------------------------------
    def _advance_entities(self) -> float:
        """Enemies patrol: bounce off walls / platform edges."""
        for e in self._entities:
            if not e.alive or e.etype != "enemy":
                continue
            nx = e.x + e.dx
            # Bounce off walls / solid / edge of platform
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

        # Check collision after move
        for e in self._entities:
            if e.alive and e.x == self._agent_x and e.y == self._agent_y:
                self._message = "Hit by an enemy!"
                self._entity_terminated = True
        return 0.0

    # ------------------------------------------------------------------
    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        state = "grounded" if self._on_ground else "airborne"
        extra = (
            f"Stars: {self._stars_collected}/{self._total_stars}"
            f"  State: {state}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    # ------------------------------------------------------------------
    def _symbol_meaning(self, ch: str) -> str:
        m: dict[str, str] = {
            "\u00b7": "empty",
            "\u25ac": "platform/ground",
            "\u2588": "wall",
            "*": "star (+1)",
            "G": "goal (top, +5)",
            "@": "you",
        }
        return m.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Climb upward by jumping between platforms. Collect stars (*) "
            "for +1 each. Reach the goal (G) at the top for +5. "
            "Avoid enemies (E) that patrol platforms."
        )
