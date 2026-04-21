"""MiniHack Quaff skill tasks."""

from __future__ import annotations

from glyphbench.envs.minihack.base import MiniHackBase
from glyphbench.envs.minihack.creatures import KOBOLD
from glyphbench.envs.minihack.items import (
    POTION_HEALING,
    SCROLL_LIGHT,
    Item,
)


class _QuaffBase(MiniHackBase):
    _distract: bool = False

    def _generate_level(self, seed: int) -> None:
        self._init_grid(7, 7)
        self._player_hp = 3  # start near death
        self._place_player(1, 1)
        self._place_stairs(5, 5)
        # Potion on the floor
        self._place_item(2, 2, POTION_HEALING)
        # Monster guarding stairs
        self._spawn_creature(KOBOLD, 4, 4)
        if self._distract:
            self._place_item(1, 4, SCROLL_LIGHT)

    def _on_quaff_potion(self, potion: Item) -> None:
        if potion.name == POTION_HEALING.name:
            self._player_hp = self._player_max_hp
            self._message += " You feel much better!"

    def _task_description(self) -> str:
        return (
            "You are near death with a monster guarding the stairs. "
            "Pick up the potion of healing (!) and QUAFF it to restore HP, "
            "then fight the monster and reach the stairs (⇣). "
            "Reward: +1 stairs, -1 death."
        )


class MiniHackQuaffEnv(_QuaffBase):
    """MiniHack Quaff: drink a potion of healing to survive combat."""

    def env_id(self) -> str:
        return "glyphbench/minihack-quaff-v0"


class MiniHackQuaffDistractEnv(_QuaffBase):
    """MiniHack Quaff (Distract): drink potion with distracting items."""

    _distract = True

    def env_id(self) -> str:
        return "glyphbench/minihack-quaff-distract-v0"
