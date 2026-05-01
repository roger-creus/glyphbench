"""Procgen Miner environment.

Dig-down game. Agent digs through dirt, collects diamonds, avoids
falling boulders, and finds the exit.

Gym ID: glyphbench/procgen-miner-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation
from glyphbench.envs.procgen.base import ProcgenBase


class MinerEnv(ProcgenBase):
    """Miner: dig through dirt, collect diamonds, avoid boulders.

    World: 20 wide x 20 tall.  View: 20 x 20 (full view).
    No gravity on agent (free movement through dug-out cells).
    Boulders (R) fall when unsupported.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN"),
        descriptions=(
            "do nothing this step",
            "move/dig one cell left",
            "move/dig one cell right",
            "move/dig one cell up",
            "move/dig one cell down",
        ),
    )
    noop_action_name = "NOOP"

    # Reward shaping (Pattern D): +0.8 across diamonds, +0.2 on exit, -1.0 on
    # terminal boulder death. Cumulative range: [-1.0, +1.0].
    _DIAMOND_BUDGET = 0.8
    _GOAL_REWARD = 0.2
    _DEATH_PENALTY = -1.0

    def __init__(self, max_turns: int = 512) -> None:
        super().__init__(max_turns=max_turns)
        self._has_gravity = False
        self._view_w = 20
        self._view_h = 20
        self._diamonds_collected: int = 0
        self._total_diamonds: int = 0
        self._alive: bool = True

    def env_id(self) -> str:
        return "glyphbench/procgen-miner-v0"

    # ------------------------------------------------------------------
    def _generate_level(self, seed: int) -> None:
        W, H = 20, 20
        self._init_world(W, H, fill="\u00b7")
        self._alive = True
        self._diamonds_collected = 0
        self._total_diamonds = 0

        # Border walls
        for x in range(W):
            self._set_cell(x, 0, "\u2588")
            self._set_cell(x, H - 1, "\u2588")
        for y in range(H):
            self._set_cell(0, y, "\u2588")
            self._set_cell(W - 1, y, "\u2588")

        # Fill interior with dirt
        for y in range(2, H - 1):
            for x in range(1, W - 1):
                self._set_cell(x, y, "\u00b7")

        # Scatter dirt, diamonds, boulders
        for y in range(2, H - 1):
            for x in range(1, W - 1):
                r = float(self.rng.random())
                if r < 0.45:
                    self._set_cell(x, y, "d")  # dirt
                elif r < 0.52:
                    self._set_cell(x, y, "D")  # diamond
                    self._total_diamonds += 1
                elif r < 0.58:
                    self._set_cell(x, y, "R")  # boulder

        # Clear starting area (top-left interior)
        for y in range(1, 3):
            for x in range(1, 4):
                self._set_cell(x, y, "\u00b7")

        # Place exit at bottom-right
        self._set_cell(W - 2, H - 2, "G")

        # Agent starts top-left
        self._agent_x = 1
        self._agent_y = 1

    # ------------------------------------------------------------------
    def _try_move(self, dx: int, dy: int) -> bool:
        """Override: agent digs through dirt and collects diamonds."""
        if dx != 0 or dy != 0:
            self._agent_dir = (dx, dy)
        nx, ny = self._agent_x + dx, self._agent_y + dy
        ch = self._world_at(nx, ny)

        if ch == "\u2588":
            return False  # can't move through walls
        if ch == "R":
            return False  # can't push boulders (could extend later)

        # Dig dirt or collect diamond
        if ch == "d":
            self._set_cell(nx, ny, "\u00b7")
        elif ch == "D":
            self._set_cell(nx, ny, "\u00b7")
            self._diamonds_collected += 1
            self._message = "Collected a diamond!"

        self._agent_x = nx
        self._agent_y = ny
        return True

    # ------------------------------------------------------------------
    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        old_diamonds = self._diamonds_collected

        if action_name == "LEFT":
            self._try_move(-1, 0)
        elif action_name == "RIGHT":
            self._try_move(1, 0)
        elif action_name == "UP":
            self._try_move(0, -1)
        elif action_name == "DOWN":
            self._try_move(0, 1)

        # Diamond reward (Pattern D progress: +0.8 distributed across all
        # diamonds the level happens to spawn).
        if self._diamonds_collected > old_diamonds:
            reward += self._DIAMOND_BUDGET / max(1, self._total_diamonds)

        # Check goal
        ch = self._world_at(self._agent_x, self._agent_y)
        if ch == "G":
            self._message = "Found the exit!"
            return reward + self._GOAL_REWARD, True, {}

        # Boulder physics: boulders fall when cell below is empty
        self._process_boulders()

        # Check if boulder fell on agent (terminal failure -> -1.0).
        if self._world_at(self._agent_x, self._agent_y) == "R":
            self._message = "Crushed by a boulder!"
            self._alive = False
            return self._DEATH_PENALTY, True, {"killed_by": "boulder"}

        return reward, False, {}

    # ------------------------------------------------------------------
    def _process_boulders(self) -> None:
        """Scan grid for unsupported boulders and make them fall."""
        # Process from bottom to top so cascading works in one pass
        for y in range(self._world_h - 2, 0, -1):
            for x in range(1, self._world_w - 1):
                if self._world_at(x, y) == "R":
                    below = self._world_at(x, y + 1)
                    if below == "\u00b7":
                        self._set_cell(x, y, "\u00b7")
                        self._set_cell(x, y + 1, "R")

    # ------------------------------------------------------------------
    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        extra = (
            f"Diamonds: {self._diamonds_collected}"
            f"/{self._total_diamonds}"
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
            "\u2588": "wall",
            "d": "dirt (diggable)",
            "D": "diamond (collect for partial reward)",
            "R": "boulder (falls, deadly: terminal -1.0)",
            "G": "exit (finishes the level)",
            "@": "you",
        }
        return m.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Dig through dirt (d) to navigate the mine. Collect every "
            "diamond (D) and reach the exit (G) — diamonds yield +0.8 "
            "total and the exit yields +0.2 (best case +1.0). Boulders (R) "
            "fall when unsupported; getting crushed ends the episode at -1.0."
        )
