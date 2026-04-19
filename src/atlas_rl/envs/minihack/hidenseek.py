"""MiniHack HideNSeek environments. Navigate corridors, avoid/find obstacles."""

from __future__ import annotations

from atlas_rl.envs.minihack.base import MiniHackBase
from atlas_rl.envs.minihack.creatures import KOBOLD


class _HideNSeekBase(MiniHackBase):
    _size: int = 9
    _has_lava: bool = False

    def _generate_level(self, seed: int) -> None:
        s = self._size
        self._init_grid(s, s)

        # Create corridor-like structure with internal walls
        for i in range(2, s - 2, 3):
            for y in range(1, s - 1):
                if y != s // 2:  # leave gap for passage
                    self._place_wall(i, y)

        # Add lava in some corridors
        if self._has_lava:
            for y in [2, s - 3]:
                for x in range(1, s - 1):
                    if self._grid[y][x] == "." and self.rng.random() < 0.3:
                        self._place_lava(x, y)

        # Player at start
        self._place_player(1, 1)
        # Stairs at end
        self._place_stairs(s - 2, s - 2)
        # Spawn a monster
        self._spawn_creature(KOBOLD, s // 2, s // 2)

    def _task_description(self) -> str:
        parts = ["Navigate through corridors to reach the stairs (>)."]
        if self._has_lava:
            parts.append("Watch out for lava (}) -- stepping in it is fatal.")
        parts.append("A monster lurks in the corridors.")
        parts.append("Reward: +1 on reaching stairs, -1 on death.")
        return " ".join(parts)


class MiniHackHideNSeekEnv(_HideNSeekBase):
    _size = 9

    def env_id(self) -> str:
        return "atlas_rl/minihack-hidenseek-v0"


class MiniHackHideNSeekMappedEnv(_HideNSeekBase):
    _size = 9

    def env_id(self) -> str:
        return "atlas_rl/minihack-hidenseek-mapped-v0"


class MiniHackHideNSeekLavaEnv(_HideNSeekBase):
    _size = 9
    _has_lava = True

    def env_id(self) -> str:
        return "atlas_rl/minihack-hidenseek-lava-v0"


class MiniHackHideNSeekBigEnv(_HideNSeekBase):
    _size = 15

    def env_id(self) -> str:
        return "atlas_rl/minihack-hidenseek-big-v0"
