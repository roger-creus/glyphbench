"""MiniHack Read skill tasks."""

from __future__ import annotations

from glyphbench.envs.minihack.base import MiniHackBase
from glyphbench.envs.minihack.items import (
    APPLE,
    POTION_SPEED,
    SCROLL_TELEPORT,
    Item,
)


class _ReadBase(MiniHackBase):
    _distract: bool = False

    def _generate_level(self, seed: int) -> None:
        self._init_grid(7, 7)
        self._place_player(1, 1)
        self._place_stairs(5, 5)
        # Block direct path with walls
        for y in range(1, 6):
            self._place_wall(3, y)
        # Scroll on the floor
        self._place_item(2, 2, SCROLL_TELEPORT)
        if self._distract:
            self._place_item(1, 3, APPLE)
            self._place_item(2, 4, POTION_SPEED)

    def _on_read_scroll(self, scroll: Item) -> None:
        if scroll.name == SCROLL_TELEPORT.name:
            # Teleport player to the stairs
            if self._goal_pos is not None:
                self._player_pos = self._goal_pos
            self._message += " You feel a wrenching sensation!"

    def _task_description(self) -> str:
        return (
            "A wall blocks your path to the stairs (⇣). "
            "Pick up the scroll of teleportation (?) and READ it to teleport "
            "past the wall. Reward: +1 stairs, -1 death."
        )


class MiniHackReadEnv(_ReadBase):
    """MiniHack Read: read a scroll of teleportation to bypass a wall."""

    def env_id(self) -> str:
        return "glyphbench/minihack-read-v0"


class MiniHackReadDistractEnv(_ReadBase):
    """MiniHack Read (Distract): read scroll with distracting items."""

    _distract = True

    def env_id(self) -> str:
        return "glyphbench/minihack-read-distract-v0"
