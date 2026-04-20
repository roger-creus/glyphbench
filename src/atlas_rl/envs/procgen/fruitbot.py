"""Procgen FruitBot environment.

Vertical auto-scroller. Agent falls through a tunnel collecting fruit
and dodging obstacles. Level scrolls downward automatically.

Gym ID: atlas_rl/procgen-fruitbot-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec
from atlas_rl.core.observation import GridObservation
from atlas_rl.envs.procgen.base import ProcgenBase


class FruitBotEnv(ProcgenBase):
    """FruitBot: vertical auto-scroller collecting fruit.

    World: 14 wide x 40 tall.  View: 14 x 20.
    Agent auto-falls 1 cell per step. Collect fruit (%) for +1,
    avoid obstacles (x) for -1 penalty.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "DOWN"),
        descriptions=(
            "do nothing (still falls 1 cell)",
            "move one cell left while falling",
            "move one cell right while falling",
            "fall 2 cells instead of 1",
        ),
    )
    noop_action_name = "NOOP"

    def __init__(self, max_turns: int = 512) -> None:
        super().__init__(max_turns=max_turns)
        self._has_gravity = False  # we manage falling ourselves
        self._view_w = 14
        self._view_h = 20
        self._fruits_collected: int = 0
        self._obstacles_hit: int = 0

    def env_id(self) -> str:
        return "atlas_rl/procgen-fruitbot-v0"

    # ------------------------------------------------------------------
    def _generate_level(self, seed: int) -> None:
        W, H = 14, 40
        self._init_world(W, H, fill=".")
        self._fruits_collected = 0
        self._obstacles_hit = 0

        # Walls on sides
        for y in range(H):
            self._set_cell(0, y, "#")
            self._set_cell(W - 1, y, "#")

        # Scatter fruit and obstacles throughout the tunnel
        for y in range(3, H - 1):
            n_items = int(self.rng.integers(0, 3))
            for _ in range(n_items):
                ix = int(self.rng.integers(1, W - 1))
                if self._world_at(ix, y) == ".":
                    if self.rng.random() < 0.6:
                        self._set_cell(ix, y, "%")
                    else:
                        self._set_cell(ix, y, "x")

            # Occasional internal walls to make it interesting
            if self.rng.random() < 0.15:
                wall_x = int(self.rng.integers(2, W - 2))
                wall_len = int(self.rng.integers(2, 5))
                for dx in range(wall_len):
                    wx = wall_x + dx
                    if 1 <= wx < W - 1:
                        self._set_cell(wx, y, "#")

        # Agent starts at top center
        self._agent_x = W // 2
        self._agent_y = 1

    # ------------------------------------------------------------------
    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0

        # Horizontal movement
        if action_name == "LEFT":
            self._try_move(-1, 0)
        elif action_name == "RIGHT":
            self._try_move(1, 0)
        elif action_name == "DOWN":
            self._agent_dir = (0, 1)

        # Fall
        fall_dist = 2 if action_name == "DOWN" else 1
        for _ in range(fall_dist):
            ny = self._agent_y + 1
            if ny >= self._world_h:
                # Reached bottom
                self._message = "Reached the bottom!"
                return reward, True, {}
            if self._is_solid(self._agent_x, ny):
                # Blocked by wall below — end
                self._message = "Blocked!"
                return reward, True, {}
            self._agent_y = ny

            # Check what we landed on
            ch = self._world_at(self._agent_x, self._agent_y)
            if ch == "%":
                self._set_cell(self._agent_x, self._agent_y, ".")
                reward += 1.0
                self._fruits_collected += 1
            elif ch == "x":
                self._set_cell(self._agent_x, self._agent_y, ".")
                reward -= 1.0
                self._obstacles_hit += 1

        # Reached bottom
        if self._agent_y >= self._world_h - 1:
            self._message = "Reached the bottom!"
            return reward, True, {}

        return reward, False, {}

    # ------------------------------------------------------------------
    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        extra = (
            f"Fruits: {self._fruits_collected}"
            f"  Obstacles hit: {self._obstacles_hit}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    # ------------------------------------------------------------------
    def _symbol_meaning(self, ch: str) -> str:
        m: dict[str, str] = {
            ".": "empty",
            "#": "wall",
            "%": "fruit (+1)",
            "x": "obstacle (-1)",
            "@": "you",
        }
        return m.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "You are a FruitBot falling through a tunnel. Collect fruit "
            "(%) for +1 each. Avoid obstacles (x) which cost -1. "
            "You fall 1 cell per step automatically. Use LEFT/RIGHT to "
            "dodge, DOWN to fall faster."
        )
