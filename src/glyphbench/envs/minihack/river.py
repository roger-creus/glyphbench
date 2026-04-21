"""MiniHack River environments. Cross a river of water or lava."""

from __future__ import annotations

from glyphbench.envs.minihack.base import MiniHackBase
from glyphbench.envs.minihack.creatures import KOBOLD, ORC


class _RiverBase(MiniHackBase):
    _grid_size: tuple[int, int] = (11, 7)
    _river_type: str = "water"  # "water" or "lava"
    _num_stones: int = 3
    _has_monsters: bool = False

    def _generate_level(self, seed: int) -> None:
        w, h = self._grid_size
        self._init_grid(w, h)

        river_y = h // 2

        # Fill river row
        place_fn = self._place_water if self._river_type == "water" else self._place_lava
        for x in range(1, w - 1):
            place_fn(x, river_y)

        # Stepping stones (floor tiles in the river)
        stone_positions: set[int] = set()
        while len(stone_positions) < self._num_stones:
            sx = int(self.rng.integers(1, w - 1))
            stone_positions.add(sx)
        for sx in stone_positions:
            self._grid[river_y][sx] = "·"

        # Player above river
        px = int(self.rng.integers(1, w - 1))
        self._place_player(px, 1)

        # Stairs below river
        stx = int(self.rng.integers(1, w - 1))
        self._place_stairs(stx, h - 2)

        # Monsters below river
        if self._has_monsters:
            for i in range(2):
                while True:
                    mx = int(self.rng.integers(1, w - 1))
                    my = int(self.rng.integers(river_y + 1, h - 1))
                    if self._grid[my][mx] == "·" and (mx, my) != (stx, h - 2):
                        break
                ctype = [KOBOLD, ORC][i % 2]
                self._spawn_creature(ctype, mx, my)

    def _task_description(self) -> str:
        danger = "lava (instant death)" if self._river_type == "lava" else "water"
        parts = [f"Cross a river of {danger} using stepping stones to reach the stairs (⇣)."]
        if self._has_monsters:
            parts.append("Hostile monsters wait on the far side.")
        parts.append("Reward: +1 on reaching stairs, -1 on death.")
        return " ".join(parts)


class MiniHackRiverEnv(_RiverBase):
    _river_type = "water"
    _num_stones = 3

    def env_id(self) -> str:
        return "glyphbench/minihack-river-v0"


class MiniHackRiverNarrowEnv(_RiverBase):
    _river_type = "water"
    _num_stones = 1

    def env_id(self) -> str:
        return "glyphbench/minihack-river-narrow-v0"


class MiniHackRiverMonsterEnv(_RiverBase):
    _river_type = "water"
    _num_stones = 3
    _has_monsters = True

    def env_id(self) -> str:
        return "glyphbench/minihack-river-monster-v0"


class MiniHackRiverLavaEnv(_RiverBase):
    _river_type = "lava"
    _num_stones = 3

    def env_id(self) -> str:
        return "glyphbench/minihack-river-lava-v0"


class MiniHackRiverMonsterLavaEnv(_RiverBase):
    _river_type = "lava"
    _num_stones = 3
    _has_monsters = True

    def env_id(self) -> str:
        return "glyphbench/minihack-river-monsterlava-v0"
