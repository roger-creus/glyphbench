"""MiniHack Wield skill tasks."""

from __future__ import annotations

from typing import Any

from atlas_rl.core.observation import GridObservation
from atlas_rl.envs.minihack.base import MOVE_VECTORS, MiniHackBase
from atlas_rl.envs.minihack.creatures import KOBOLD
from atlas_rl.envs.minihack.items import POTION_SPEED, SCROLL_LIGHT, SWORD


class _WieldBase(MiniHackBase):
    _distract: bool = False

    def _generate_level(self, seed: int) -> None:
        self._init_grid(7, 7)
        self._place_player(1, 1)
        self._place_stairs(5, 5)
        # Weapon on the floor
        self._place_item(2, 1, SWORD)
        # Monsters guarding stairs
        self._spawn_creature(KOBOLD, 4, 3)
        self._spawn_creature(KOBOLD, 4, 4)
        if self._distract:
            self._place_item(1, 4, SCROLL_LIGHT)
            self._place_item(3, 1, POTION_SPEED)

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]

        # Override melee to use wielded weapon damage
        if name in MOVE_VECTORS:
            dx, dy = MOVE_VECTORS[name]
            nx, ny = self._player_pos[0] + dx, self._player_pos[1] + dy
            monster = self._creature_at(nx, ny)
            if monster is not None and self._wielding is not None:
                # Wielded weapon does more damage (6-10 instead of 1-4)
                dmg = max(1, int(self.rng.integers(6, 11)))
                monster.hp -= dmg
                self._message = f"You slash the {monster.ctype.name} with your {self._wielding.name}!"
                if monster.hp <= 0:
                    self._message += f" The {monster.ctype.name} dies."
                    self._creatures = [c for c in self._creatures if c.hp > 0]

                # Monster turns
                self._move_monsters()
                if self._player_hp <= 0:
                    self._message = (self._message + " You die.").strip()
                    info: dict[str, Any] = {"cause_of_death": "monster"}
                    return (
                        self._render_current_observation(),
                        -1.0,
                        True,
                        False,
                        info,
                    )

                # Check goal
                if self._goal_pos and self._player_pos == self._goal_pos:
                    self._message = "You reach the stairs. You descend."
                    return (
                        self._render_current_observation(),
                        1.0,
                        True,
                        False,
                        {"goal_reached": True, "player_pos": self._player_pos, "hp": self._player_hp},
                    )

                return (
                    self._render_current_observation(),
                    0.0,
                    False,
                    False,
                    {"player_pos": self._player_pos, "hp": self._player_hp},
                )

        # Fall through to base for all other actions / unarmed combat
        return super()._step(action)

    def _task_description(self) -> str:
        return (
            "Monsters guard the stairs. Pick up the sword ()) and WIELD it "
            "for much higher damage, then fight through and reach the stairs (>). "
            "Reward: +1 stairs, -1 death."
        )


class MiniHackWieldEnv(_WieldBase):
    """MiniHack Wield: wield a weapon for higher combat damage."""

    def env_id(self) -> str:
        return "atlas_rl/minihack-wield-v0"


class MiniHackWieldDistractEnv(_WieldBase):
    """MiniHack Wield (Distract): wield weapon with distracting items."""

    _distract = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-wield-distract-v0"
