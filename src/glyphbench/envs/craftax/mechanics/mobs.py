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

import random
from enum import Enum
from typing import Callable, TypedDict

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


class MobLike(TypedDict):
    """Structural type used by AI helpers. Matches the env-level Mob TypedDict
    plus the attack_cooldown field added by T20."""
    type: str
    x: int
    y: int
    hp: int
    max_hp: int
    is_boss: bool
    floor: int
    attack_cooldown: int


MELEE_ATTACK_COOLDOWN: int = 5
RANGED_ATTACK_COOLDOWN: int = 4
MOB_DESPAWN_DISTANCE: int = 14


def step_melee_mob(
    mob: MobLike,
    *,
    player_x: int,
    player_y: int,
    is_blocked_for_mob: Callable[[int, int], bool],
    apply_damage_to_player: Callable[[int], None],
    rng: random.Random,
    is_fighting_boss: bool,
    damage_for_mob: Callable[[MobLike], int],
) -> None:
    """Advance one melee mob for one tick. Mirrors upstream
    _move_melee_mob (game_logic.py:1100-1291) at phase-α fidelity:

    - If adjacent (Manhattan == 1) and cooldown <= 0: attack and reset
      cooldown to MELEE_ATTACK_COOLDOWN. No movement this tick.
    - Otherwise tick the cooldown down by 1 (clamped at 0).
    - Movement: 75% chase if dist_sum < 10 OR is_fighting_boss; else
      random walk over (+-1, 0) / (0, +-1) / stay.

    Damage value is supplied by `damage_for_mob` so the caller controls
    the per-mob damage table. Movement is bounded by `is_blocked_for_mob`.
    """
    # Tick cooldown down before considering attack.
    if mob["attack_cooldown"] > 0:
        mob["attack_cooldown"] -= 1
        # Even with cooldown ticking, the mob still moves this tick (matches
        # upstream where cooldown does not block movement).
    else:
        manhattan = abs(mob["x"] - player_x) + abs(mob["y"] - player_y)
        if manhattan == 1:
            apply_damage_to_player(damage_for_mob(mob))
            mob["attack_cooldown"] = MELEE_ATTACK_COOLDOWN
            return  # attacking mobs do not also move this tick

    # Movement.
    manhattan = abs(mob["x"] - player_x) + abs(mob["y"] - player_y)
    chase = is_fighting_boss or (manhattan < 10 and rng.random() < 0.75)
    if chase:
        dist_x = player_x - mob["x"]
        dist_y = player_y - mob["y"]
        if abs(dist_x) >= abs(dist_y):
            dx = 1 if dist_x > 0 else (-1 if dist_x < 0 else 0)
            dy = 0
        else:
            dx = 0
            dy = 1 if dist_y > 0 else (-1 if dist_y < 0 else 0)
    else:
        dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)])
    nx, ny = mob["x"] + dx, mob["y"] + dy
    if not is_blocked_for_mob(nx, ny):
        mob["x"], mob["y"] = nx, ny


def should_despawn(
    mob: MobLike,
    *,
    player_x: int,
    player_y: int,
    is_fighting_boss: bool,
) -> bool:
    """Return True if this mob should be removed this tick.

    Upstream rule (game_logic.py:1100-1291 + 1391-1608):
    - Manhattan distance from player >= MOB_DESPAWN_DISTANCE (=14).
    - Boss-fight melee + ranged mobs are exempt so the necromancer
      summon wave persists.

    Passive mobs (cow, snail, bat) follow the same despawn rule even
    in boss fights upstream — but the boss floor has no passives, so
    the distinction doesn't bite in practice.
    """
    if is_fighting_boss and mob["type"] in MELEE_MOB_NAMES + RANGED_MOB_NAMES + (
        "zombie", "skeleton", "skeleton_archer",  # legacy names still in use
    ):
        return False
    dist_sum = abs(mob["x"] - player_x) + abs(mob["y"] - player_y)
    return dist_sum >= MOB_DESPAWN_DISTANCE
