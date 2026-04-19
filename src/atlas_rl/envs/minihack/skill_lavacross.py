"""MiniHack LavaCross skill tasks."""

from __future__ import annotations

from typing import Any

from atlas_rl.core.observation import GridObservation
from atlas_rl.envs.minihack.base import MiniHackBase
from atlas_rl.envs.minihack.items import POTION_LEVITATION, RING_LEVITATION, Item


class _LavaCrossBase(MiniHackBase):
    _potion_on_floor: bool = True
    _ring_on_floor: bool = True
    _potion_in_inv: bool = False
    _ring_in_inv: bool = False

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._levitating_turns: int = 0

    def _generate_level(self, seed: int) -> None:
        self._init_grid(11, 7)
        self._levitating_turns = 0
        self._place_player(1, 3)
        self._place_stairs(9, 3)

        # Lava pit in the middle (x=4 to x=6)
        for x in range(4, 7):
            for y in range(1, 6):
                self._place_lava(x, y)

        # Place levitation items
        if self._potion_on_floor:
            self._place_item(2, 3, POTION_LEVITATION)
        if self._ring_on_floor:
            self._place_item(2, 1, RING_LEVITATION)
        if self._potion_in_inv:
            self._inventory.append(POTION_LEVITATION)
        if self._ring_in_inv:
            self._inventory.append(RING_LEVITATION)

    def _on_quaff_potion(self, potion: Item) -> None:
        if potion.name == "potion of levitation":
            self._levitating_turns = 20
            self._message += " You start to float!"

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        # While levitating, temporarily convert lava to safe floor so the
        # base _step terrain check doesn't kill the player.
        lava_cells: list[tuple[int, int]] = []
        if self._levitating_turns > 0:
            for y in range(self._grid_h):
                for x in range(self._grid_w):
                    if self._grid[y][x] == "}":
                        lava_cells.append((x, y))
                        self._grid[y][x] = "."

        obs, reward, terminated, truncated, info = super()._step(action)

        # Restore lava tiles
        for x, y in lava_cells:
            self._grid[y][x] = "}"

        # Decrement levitation
        if self._levitating_turns > 0:
            self._levitating_turns -= 1
            if self._levitating_turns == 0:
                self._message += " You float gently to the ground."
                # Re-render to include updated message
                obs = self._render_current_observation()

        return obs, reward, terminated, truncated, info

    def _task_description(self) -> str:
        return (
            "A lava pit (}) blocks your path to the stairs (>). "
            "Use a potion of levitation (QUAFF) or ring of levitation to float over it. "
            "Stepping on lava without levitation is fatal. "
            "Reward: +1 stairs, -1 death."
        )


class MiniHackLavaCrossFullEnv(_LavaCrossBase):
    _potion_on_floor = True
    _ring_on_floor = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-lavacross-full-v0"


class MiniHackLavaCrossLevitateEnv(_LavaCrossBase):
    _potion_on_floor = True
    _ring_on_floor = False

    def env_id(self) -> str:
        return "atlas_rl/minihack-lavacross-levitate-v0"


class MiniHackLavaCrossPotionInvEnv(_LavaCrossBase):
    _potion_on_floor = False
    _ring_on_floor = False
    _potion_in_inv = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-lavacross-levitate-potion-inv-v0"


class MiniHackLavaCrossRingInvEnv(_LavaCrossBase):
    _potion_on_floor = False
    _ring_on_floor = False
    _ring_in_inv = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-lavacross-levitate-ring-inv-v0"
