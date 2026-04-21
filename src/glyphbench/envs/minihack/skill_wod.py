"""MiniHack Wand of Death skill tasks."""

from __future__ import annotations

from glyphbench.envs.minihack.base import MiniHackBase
from glyphbench.envs.minihack.creatures import KOBOLD, OGRE, ORC, ZOMBIE
from glyphbench.envs.minihack.items import WAND_COLD, WAND_DEATH, WAND_FIRE, Item


class _WoDBase(MiniHackBase):
    _num_monsters: int = 1
    _has_distractors: bool = False
    _is_dark: bool = False

    def _generate_level(self, seed: int) -> None:
        self._init_grid(9, 9)
        self._dark = self._is_dark
        self._place_player(1, 4)
        self._place_stairs(7, 4)

        # Place wand of death
        wx = int(self.rng.integers(2, 5))
        wy = int(self.rng.integers(1, 8))
        self._place_item(wx, wy, WAND_DEATH)

        # Distractor wands
        if self._has_distractors:
            self._place_item(3, 2, WAND_FIRE)
            self._place_item(3, 6, WAND_COLD)

        # Monsters between player and stairs
        monster_types = [KOBOLD, ORC, ZOMBIE, OGRE]
        for i in range(self._num_monsters):
            ctype = monster_types[i % len(monster_types)]
            attempts = 0
            while attempts < 50:
                mx = int(self.rng.integers(5, 7))
                my = int(self.rng.integers(1, 8))
                if self._creature_at(mx, my) is None:
                    self._spawn_creature(ctype, mx, my)
                    break
                attempts += 1

    def _on_zap_wand(self, wand: Item) -> None:
        if wand.name == "wand of death":
            # Kill nearest visible monster
            px, py = self._player_pos
            nearest = None
            nearest_dist = float("inf")
            for c in self._creatures:
                if c.hp <= 0:
                    continue
                dist = abs(c.x - px) + abs(c.y - py)
                if dist < nearest_dist:
                    nearest = c
                    nearest_dist = dist
            if nearest is not None:
                nearest.hp = 0
                self._message += f" The {nearest.ctype.name} is killed!"
                self._creatures = [c for c in self._creatures if c.hp > 0]

    def _task_description(self) -> str:
        return (
            f"A room with {self._num_monsters} hostile monster(s). "
            f"Find the wand of death (/) on the floor, pick it up with PICKUP, "
            f"then use ZAP to kill monsters. "
            f"Reach the stairs (⇣). Reward: +1 stairs, -1 death."
        )


class MiniHackWoDEasyEnv(_WoDBase):
    _num_monsters = 1

    def env_id(self) -> str:
        return "glyphbench/minihack-wod-easy-v0"


class MiniHackWoDMediumEnv(_WoDBase):
    _num_monsters = 2

    def env_id(self) -> str:
        return "glyphbench/minihack-wod-medium-v0"


class MiniHackWoDHardEnv(_WoDBase):
    _num_monsters = 3
    _has_distractors = True

    def env_id(self) -> str:
        return "glyphbench/minihack-wod-hard-v0"


class MiniHackWoDProEnv(_WoDBase):
    _num_monsters = 4
    _has_distractors = True
    _is_dark = True

    def env_id(self) -> str:
        return "glyphbench/minihack-wod-pro-v0"
