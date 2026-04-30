"""Phase-α mob roster + AI primitives.

Mirrors upstream:
- constants.py:118-123 — MobType enum
- constants.py:138-152 — FLOOR_MOB_MAPPING
- constants.py:155-169 — FLOOR_MOB_SPAWN_CHANCE
- constants.py:178-255 — MOB_TYPE_COLLISION_MAPPING (deferred to phase β/γ)

This module is DATA-ONLY in T18. AI primitives (step_melee_mob, step_ranged_mob,
should_despawn) are added in T21, T23, T24.
"""
from __future__ import annotations

from enum import Enum

from glyphbench.envs.craftax.mechanics.projectiles import ProjectileType


class MobType(Enum):
    """Upstream constants.py:118-123."""
    PASSIVE = 0
    MELEE = 1
    RANGED = 2
    PROJECTILE = 3


PASSIVE_MOB_NAMES: tuple[str, ...] = ("cow", "bat", "snail")

MELEE_MOB_NAMES: tuple[str, ...] = (
    "zombie", "gnome_warrior", "orc_soldier", "lizard",
    "knight", "troll", "pigman", "frost_troll",
)

RANGED_MOB_NAMES: tuple[str, ...] = (
    "skeleton", "gnome_archer", "orc_mage", "kobold",
    "knight_archer", "deep_thing", "fire_elemental", "ice_elemental",
)


# Per-floor roster: which mob name spawns in each class on each floor.
# Indices are resolved from upstream FLOOR_MOB_MAPPING (constants.py:138-152)
# using the texture-load order as the name index:
#   passive[0]=cow, [1]=bat, [2]=snail
#   melee[0]=zombie, [1]=gnome_warrior, [2]=orc_soldier, [3]=lizard,
#         [4]=knight, [5]=troll, [6]=pigman, [7]=frost_troll
#   ranged[0]=skeleton, [1]=gnome_archer, [2]=orc_mage, [3]=kobold,
#          [4]=knight_archer, [5]=deep_thing, [6]=fire_elemental, [7]=ice_elemental
FLOOR_MOB_MAPPING: tuple[dict[str, str], ...] = (
    # 0 — Overworld: [0, 0, 0]
    {"passive": "cow", "melee": "zombie", "ranged": "skeleton"},
    # 1 — Dungeon: [2, 2, 2]
    {"passive": "snail", "melee": "orc_soldier", "ranged": "orc_mage"},
    # 2 — Gnomish Mines: [1, 1, 1]
    {"passive": "bat", "melee": "gnome_warrior", "ranged": "gnome_archer"},
    # 3 — Sewers: [2, 3, 3]
    {"passive": "snail", "melee": "lizard", "ranged": "kobold"},
    # 4 — Vaults: [2, 4, 4]
    {"passive": "snail", "melee": "knight", "ranged": "knight_archer"},
    # 5 — Troll Mines: [1, 5, 5]
    {"passive": "bat", "melee": "troll", "ranged": "deep_thing"},
    # 6 — Fire Realm: [1, 6, 6]
    {"passive": "bat", "melee": "pigman", "ranged": "fire_elemental"},
    # 7 — Ice Realm: [1, 7, 7]
    {"passive": "bat", "melee": "frost_troll", "ranged": "ice_elemental"},
    # 8 — Boss: [0, 0, 0] — wave-summon defaults; phase γ adds boss progression
    {"passive": "cow", "melee": "zombie", "ranged": "skeleton"},
)


# Maps each ranged-mob name to the projectile type it fires.
# Upstream: constants.py:320-331 — RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING.
RANGED_MOB_TO_PROJECTILE: dict[str, ProjectileType] = {
    "skeleton": ProjectileType.ARROW,
    "gnome_archer": ProjectileType.ARROW,
    "orc_mage": ProjectileType.FIREBALL,
    "kobold": ProjectileType.DAGGER,
    "knight_archer": ProjectileType.ARROW2,
    "deep_thing": ProjectileType.SLIMEBALL,
    "fire_elemental": ProjectileType.FIREBALL2,
    "ice_elemental": ProjectileType.ICEBALL2,
}
