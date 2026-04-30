"""Projectile entity and container for Phase α.

Projectiles are persistent tile entities that travel one step per turn
along (dx, dy) and are removed when they hit a solid block, a target,
or leave the map. Damage is scalar in Phase α; Phase γ converts it to
the (physical, fire, ice) 3-vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class ProjectileType(Enum):
    """Upstream-mirror of constants.py:125-134."""
    ARROW = 0
    DAGGER = 1
    FIREBALL = 2
    ICEBALL = 3
    ARROW2 = 4
    SLIMEBALL = 5
    FIREBALL2 = 6
    ICEBALL2 = 7


@dataclass(slots=True)
class ProjectileEntity:
    """A projectile travelling 1 tile/turn along (dx, dy)."""
    kind: ProjectileType
    x: int
    y: int
    dx: int
    dy: int
    damage: int

    def advance(self) -> None:
        """Move one tile along the direction vector."""
        self.x += self.dx
        self.y += self.dy


def step_player_projectiles(
    projectiles: list[ProjectileEntity],
    *,
    map_w: int,
    map_h: int,
    blocked_fn: Callable[[int, int], bool],
    hit_fn: Callable[[int, int], bool],
) -> list[ProjectileEntity]:
    """Advance each projectile one tile. Drop those that:
    - go out of map bounds, OR
    - land on a blocked tile (solid block), OR
    - register a hit on a target (mob).
    Returns the surviving projectiles list.
    """
    survivors: list[ProjectileEntity] = []
    for p in projectiles:
        p.advance()
        if p.x < 0 or p.x >= map_w or p.y < 0 or p.y >= map_h:
            continue
        if blocked_fn(p.x, p.y):
            continue
        if hit_fn(p.x, p.y):
            continue
        survivors.append(p)
    return survivors
