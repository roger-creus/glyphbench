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


# ---------------------------------------------------------------------------
# T04γ: Multiplicative armour defense formula
# ---------------------------------------------------------------------------

def player_defense_vec(
    armor_slots: dict[str, int],
    armor_enchants: dict[str, int],
) -> DamageVec:
    """Compute per-element defense vector from 4-slot armour state.

    Per upstream util/game_logic_utils.py:233-249:
    - Each slot with armour (tier > 0) grants 0.1 phys defense.
    - Each slot with a fire enchant grants 0.2 fire defense.
    - Each slot with an ice enchant grants 0.2 ice defense.
    - Defense per element capped at 1.0.
    """
    phys = 0.0
    fire = 0.0
    ice = 0.0
    for slot in ("helmet", "chest", "legs", "boots"):
        if armor_slots.get(slot, 0) > 0:
            phys += 0.1
        enchant = armor_enchants.get(slot, 0)
        if enchant == 1:
            fire += 0.2
        elif enchant == 2:
            ice += 0.2
    return (min(1.0, phys), min(1.0, fire), min(1.0, ice))


def damage_dealt_to_player(
    armor_slots: dict[str, int],
    armor_enchants: dict[str, int],
    damage_vec: DamageVec,
) -> int:
    """Apply per-slot armour + enchant to incoming damage. Returns rounded int."""
    def_v = player_defense_vec(armor_slots, armor_enchants)
    net = sum(d * (1.0 - dv) for d, dv in zip(damage_vec, def_v))
    return max(0, int(round(net)))
