"""MiniHack CorridorBattle environments. Fight through monsters in a corridor."""

from __future__ import annotations

from glyphbench.envs.minihack.base import MiniHackBase
from glyphbench.envs.minihack.creatures import KOBOLD, ORC, RAT


class _CorridorBattleBase(MiniHackBase):
    _is_dark: bool = False
    _num_monsters: int = 4

    def _generate_level(self, seed: int) -> None:
        w, h = 15, 5
        self._init_grid(w, h)
        self._dark = self._is_dark

        # Fill everything with walls, carve a corridor
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if y != h // 2:
                    self._place_wall(x, y)

        corridor_y = h // 2
        self._place_player(1, corridor_y)
        self._place_stairs(w - 2, corridor_y)

        # Place monsters along the corridor
        monster_types = [RAT, KOBOLD, ORC]
        for i in range(self._num_monsters):
            ctype = monster_types[i % len(monster_types)]
            attempts = 0
            while attempts < 50:
                mx = int(self.rng.integers(3, w - 3))
                if (
                    self._grid[corridor_y][mx] == "·"
                    and (mx, corridor_y) != (1, corridor_y)
                    and self._creature_at(mx, corridor_y) is None
                ):
                    self._spawn_creature(ctype, mx, corridor_y)
                    break
                attempts += 1

    def _task_description(self) -> str:
        parts = [
            f"Fight through {self._num_monsters} monsters in a corridor "
            f"to reach the stairs (⇣)."
        ]
        if self._is_dark:
            parts.append("The corridor is dark -- you can only see adjacent tiles.")
        parts.append(
            "Move into a monster to attack it. Reward: +1 on reaching stairs, -1 on death."
        )
        return " ".join(parts)


class MiniHackCorridorBattleEnv(_CorridorBattleBase):
    def env_id(self) -> str:
        return "glyphbench/minihack-corridorbattle-v0"


class MiniHackCorridorBattleDarkEnv(_CorridorBattleBase):
    _is_dark = True

    def env_id(self) -> str:
        return "glyphbench/minihack-corridorbattle-dark-v0"
