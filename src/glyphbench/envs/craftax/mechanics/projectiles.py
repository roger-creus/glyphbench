"""Projectile entity and container for Phase α.

Projectiles are persistent tile entities that travel one step per turn
along (dx, dy) and are removed when they hit a solid block, a target,
or leave the map. Damage is scalar in Phase α; Phase γ converts it to
the (physical, fire, ice) 3-vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
