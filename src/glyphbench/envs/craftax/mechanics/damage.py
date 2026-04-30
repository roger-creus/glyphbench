"""Phase-γ damage subsystem: 3-vector damage with per-element resistance.
Damage tuple: (physical, fire, ice).
"""
from __future__ import annotations

DamageVec = tuple[float, float, float]


def damage_vec_from_projectile(kind, scalar: int) -> DamageVec:
    """Map projectile kind + scalar damage to 3-vec.

    ARROW/ARROW2/DAGGER → all phys.
    FIREBALL/FIREBALL2 → all fire.
    ICEBALL/ICEBALL2 → all ice.
    SLIMEBALL → mixed (phys/fire/ice each ~1/3).
    """
    from glyphbench.envs.craftax.mechanics.projectiles import ProjectileType
    if kind in (ProjectileType.ARROW, ProjectileType.ARROW2, ProjectileType.DAGGER):
        return (float(scalar), 0.0, 0.0)
    if kind in (ProjectileType.FIREBALL, ProjectileType.FIREBALL2):
        return (0.0, float(scalar), 0.0)
    if kind in (ProjectileType.ICEBALL, ProjectileType.ICEBALL2):
        return (0.0, 0.0, float(scalar))
    if kind == ProjectileType.SLIMEBALL:
        third = scalar / 3.0
        return (third, third, third)
    return (float(scalar), 0.0, 0.0)


def vec_sum(vec: DamageVec) -> int:
    """Round to integer total damage."""
    return int(round(sum(vec)))
