"""MiniHack Eat skill tasks."""

from __future__ import annotations

from typing import Any

from atlas_rl.core.observation import GridObservation
from atlas_rl.envs.minihack.base import MiniHackBase
from atlas_rl.envs.minihack.items import FOOD_RATION, POTION_SPEED, SCROLL_LIGHT


class _EatBase(MiniHackBase):
    _distract: bool = False

    def _generate_level(self, seed: int) -> None:
        self._init_grid(7, 7)
        self._hunger = 20  # start hungry
        self._place_player(1, 1)
        self._place_stairs(5, 5)
        # Food on the floor
        self._place_item(3, 3, FOOD_RATION)
        if self._distract:
            self._place_item(2, 4, SCROLL_LIGHT)
            self._place_item(4, 2, POTION_SPEED)

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        if not terminated:
            self._hunger -= 1
            if self._hunger <= 0:
                self._message = "You starve to death!"
                terminated = True
                reward = -1.0
                info["cause_of_death"] = "starvation"
                obs = self._render_current_observation()
        return obs, reward, terminated, truncated, info

    def _task_description(self) -> str:
        return (
            "You are starving! Find food (%) on the floor, pick it up with PICKUP, "
            "then use EAT to consume it. Then reach the stairs (>). "
            "You lose 1 hunger per turn and die at 0. Reward: +1 stairs, -1 death."
        )


class MiniHackEatEnv(_EatBase):
    """MiniHack Eat: pick up and eat food before starving."""

    def env_id(self) -> str:
        return "atlas_rl/minihack-eat-v0"


class MiniHackEatDistractEnv(_EatBase):
    """MiniHack Eat (Distract): eat food with distracting items."""

    _distract = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-eat-distract-v0"
