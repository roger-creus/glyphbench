"""MiniHack Sink skill tasks."""

from __future__ import annotations

from atlas_rl.envs.minihack.base import MiniHackBase
from atlas_rl.envs.minihack.items import POTION_SPEED, SCROLL_LIGHT


class _SinkBase(MiniHackBase):
    _distract: bool = False

    def _generate_level(self, seed: int) -> None:
        self._init_grid(7, 7)
        self._place_player(1, 1)
        self._place_stairs(5, 5)
        # Place a sink
        self._grid[3][3] = "{"
        if self._distract:
            self._place_item(2, 2, SCROLL_LIGHT)
            self._place_item(4, 4, POTION_SPEED)

    def _task_description(self) -> str:
        return (
            "A room with a sink ({). Navigate to the stairs (>). "
            "You may APPLY near the sink for a random effect. "
            "Reward: +1 on reaching stairs."
        )


class MiniHackSinkEnv(_SinkBase):
    """MiniHack Sink: room with a sink terrain."""

    def env_id(self) -> str:
        return "atlas_rl/minihack-sink-v0"


class MiniHackSinkDistractEnv(_SinkBase):
    """MiniHack Sink (Distract): sink room with distracting items."""

    _distract = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-sink-distract-v0"
