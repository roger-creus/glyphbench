"""Projectile entity and container for Phase α.

Projectiles are persistent tile entities that travel one step per turn
along (dx, dy) and are removed when they hit a solid block, a target,
or leave the map. Damage is scalar in Phase α; Phase γ T12γ adds the
optional damage_vec 3-tuple for elemental (physical, fire, ice) damage.
If damage_vec is None, scalar damage is treated as pure physical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class ProjectileEntity:
    """A projectile travelling 1 tile/turn along (dx, dy).

    damage: scalar fallback (legacy / non-elemental projectiles).
    damage_vec: optional (phys, fire, ice) 3-vec for elemental projectiles.
                If None, damage is treated as pure physical ((damage, 0, 0)).
    """
    kind: ProjectileType
    x: int
    y: int
    dx: int
    dy: int
    damage: int
    damage_vec: tuple[float, float, float] | None = field(default=None)

    def advance(self) -> None:
        """Move one tile along the direction vector."""
        self.x += self.dx
        self.y += self.dy


def step_mob_projectiles(
    projectiles: list[ProjectileEntity],
    *,
    map_w: int,
    map_h: int,
    blocked_fn: Callable[[ProjectileEntity], bool],
    block_destruction_fn: Callable[[ProjectileEntity], bool],
    hit_player_fn: Callable[[ProjectileEntity], bool],
) -> list[ProjectileEntity]:
    """Advance each mob projectile one tile. Drop those that:
    - go out of map bounds, OR
    - destroy a destructible tile (furnace / crafting table) and die on impact
      (block_destruction_fn fires BEFORE blocked_fn so destructible structures
      are not mis-classified as ordinary solid blocks and skipped), OR
    - land on a solid tile that is NOT destructible (drop without effect), OR
    - hit the player (damage applied via hit_player_fn).

    Returns the surviving projectiles list.

    Mirrors upstream _move_mob_projectile (game_logic.py:1616-1710). Sleep
    cancellation on hit happens inside hit_player_fn (the env-level closure).

    Ordering note: block_destruction_fn is checked before blocked_fn because
    furnaces/crafting tables appear in _SOLID_TILES; without this ordering,
    the blocked check would fire first and destroy logic would never run.
    """
    survivors: list[ProjectileEntity] = []
    for p in projectiles:
        p.advance()
        if p.x < 0 or p.x >= map_w or p.y < 0 or p.y >= map_h:
            continue
        if block_destruction_fn(p):
            continue  # destructible tile → projectile dies on impact
        if blocked_fn(p):
            continue
        if hit_player_fn(p):
            continue
        survivors.append(p)
    return survivors


def step_player_projectiles(
    projectiles: list[ProjectileEntity],
    *,
    map_w: int,
    map_h: int,
    blocked_fn: Callable[[ProjectileEntity], bool],
    hit_fn: Callable[[ProjectileEntity], bool],
) -> list[ProjectileEntity]:
    """Advance each projectile one tile. Drop those that:
    - go out of map bounds, OR
    - land on a blocked tile (solid block), OR
    - register a hit on a target (mob).
    Returns the surviving projectiles list.

    Phase-α scope: collision is checked only at the post-advance position.
    Upstream Craftax (game_logic.py:1786-1799) also checks the pre-advance
    tile so a stationary mob walked into by a moving projectile is hit;
    we defer that to Phase γ when projectile damage becomes 3-vector.

    blocked_fn and hit_fn both receive the ProjectileEntity (not raw x, y
    coordinates). This lets hit_fn read p.damage directly, avoiding the
    stale-lookup bug that arose when two projectiles shared a post-advance
    tile and next() resolved to the wrong one.
    """
    survivors: list[ProjectileEntity] = []
    for p in projectiles:
        p.advance()
        if p.x < 0 or p.x >= map_w or p.y < 0 or p.y >= map_h:
            continue
        if blocked_fn(p):
            continue
        if hit_fn(p):
            continue
        survivors.append(p)
    return survivors
