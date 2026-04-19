"""MiniHack Pray skill tasks."""

from __future__ import annotations

from atlas_rl.envs.minihack.base import MiniHackBase
from atlas_rl.envs.minihack.items import POTION_SPEED, SCROLL_LIGHT


class _PrayBase(MiniHackBase):
    _distract: bool = False

    def _generate_level(self, seed: int) -> None:
        self._init_grid(7, 7)
        self._player_hp = 3  # start near death
        self._place_player(1, 1)
        self._place_stairs(5, 5)
        # Block direct path with walls
        for y in range(1, 6):
            self._place_wall(3, y)
        self._grid[3][3] = "."  # gap in wall
        if self._distract:
            self._place_item(2, 2, SCROLL_LIGHT)
            self._place_item(2, 4, POTION_SPEED)

    def _on_pray(self) -> None:
        self._player_hp = self._player_max_hp
        self._message += " You feel much better!"

    def _task_description(self) -> str:
        return (
            "You are near death (low HP). Use PRAY to heal yourself, "
            "then navigate to the stairs (>). Reward: +1 stairs, -1 death."
        )


class MiniHackPrayEnv(_PrayBase):
    """MiniHack Pray: pray to heal low HP, then reach stairs."""

    def env_id(self) -> str:
        return "atlas_rl/minihack-pray-v0"


class MiniHackPrayDistractEnv(_PrayBase):
    """MiniHack Pray (Distract): pray to heal with distracting items."""

    _distract = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-pray-distract-v0"
