"""Procgen Miner environment.

Dig-down game. Agent digs through dirt, collects diamonds, avoids
falling boulders, and finds the exit.

Gym ID: atlas_rl/procgen-miner-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec
from atlas_rl.envs.procgen.base import ProcgenBase


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

    def __init__(self, max_turns: int = 512) -> None:
        super().__init__(max_turns=max_turns)
        self._has_gravity = False
        self._view_w = 20
        self._view_h = 20
        self._diamonds_collected: int = 0
        self._total_diamonds: int = 0
        self._alive: bool = True

    def env_id(self) -> str:
        return "atlas_rl/procgen-miner-v0"

    # ------------------------------------------------------------------
    def _generate_level(self, seed: int) -> None:
        W, H = 20, 20
        self._init_world(W, H, fill=".")
        self._alive = True
        self._diamonds_collected = 0
        self._total_diamonds = 0

        # Border walls
        for x in range(W):
            self._set_cell(x, 0, "#")
            self._set_cell(x, H - 1, "#")
        for y in range(H):
            self._set_cell(0, y, "#")
            self._set_cell(W - 1, y, "#")

        # Fill interior with dirt
        for y in range(2, H - 1):
            for x in range(1, W - 1):
                self._set_cell(x, y, ".")

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
                self._set_cell(x, y, ".")

        # Place exit at bottom-right
        self._set_cell(W - 2, H - 2, "G")

        # Agent starts top-left
        self._agent_x = 1
        self._agent_y = 1

    # ------------------------------------------------------------------
    def _try_move(self, dx: int, dy: int) -> bool:
        """Override: agent digs through dirt and collects diamonds."""
        nx, ny = self._agent_x + dx, self._agent_y + dy
        ch = self._world_at(nx, ny)

        if ch == "#":
            return False  # can't move through walls
        if ch == "R":
            return False  # can't push boulders (could extend later)

        # Dig dirt or collect diamond
        if ch == "d":
            self._set_cell(nx, ny, ".")
        elif ch == "D":
            self._set_cell(nx, ny, ".")
            self._diamonds_collected += 1
            self._score += 1.0
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

        # Diamond reward
        if self._diamonds_collected > old_diamonds:
            reward += 1.0

        # Check goal
        ch = self._world_at(self._agent_x, self._agent_y)
        if ch == "G":
            self._message = "Found the exit!"
            return reward + 10.0, True, {}

        # Boulder physics: boulders fall when cell below is empty
        self._process_boulders()

        # Check if boulder fell on agent
        if self._world_at(self._agent_x, self._agent_y) == "R":
            self._message = "Crushed by a boulder!"
            self._alive = False
            return 0.0, True, {"killed_by": "boulder"}

        return reward, False, {}

    # ------------------------------------------------------------------
    def _process_boulders(self) -> None:
        """Scan grid for unsupported boulders and make them fall."""
        # Process from bottom to top so cascading works in one pass
        for y in range(self._world_h - 2, 0, -1):
            for x in range(1, self._world_w - 1):
                if self._world_at(x, y) == "R":
                    below = self._world_at(x, y + 1)
                    if below == ".":
                        self._set_cell(x, y, ".")
                        self._set_cell(x, y + 1, "R")

    # ------------------------------------------------------------------
    def _symbol_meaning(self, ch: str) -> str:
        m: dict[str, str] = {
            ".": "empty",
            "#": "wall",
            "d": "dirt (diggable)",
            "D": "diamond (+1)",
            "R": "boulder (falls, deadly)",
            "G": "exit (+10)",
            "@": "you",
        }
        return m.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Dig through dirt (d) to navigate the mine. Collect diamonds "
            "(D) for +1 each. Reach the exit (G) for +10. Beware of "
            "boulders (R) — they fall when unsupported and crush you."
        )
