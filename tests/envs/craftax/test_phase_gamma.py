"""Phase-γ T01+T02+T03+T04 tests:
3-vector damage scaffold + per-mob defense + 4-slot armour state + multiplicative defense.
"""
from __future__ import annotations

import pytest

from glyphbench.envs.craftax.mechanics.damage import (
    damage_vec_from_projectile,
    vec_sum,
    player_defense_vec,
    damage_dealt_to_player,
)
from glyphbench.envs.craftax.mechanics.mobs import MOB_DEFENSE_VEC, damage_dealt_to_mob
from glyphbench.envs.craftax.mechanics.projectiles import ProjectileType


def test_arrow_damage_vec():
    assert damage_vec_from_projectile(ProjectileType.ARROW, 2) == (2.0, 0.0, 0.0)


def test_fireball_damage_vec():
    assert damage_vec_from_projectile(ProjectileType.FIREBALL, 4) == (0.0, 4.0, 0.0)


def test_slimeball_damage_vec():
    vec = damage_vec_from_projectile(ProjectileType.SLIMEBALL, 6)
    assert len(vec) == 3
    assert all(abs(c - 2.0) < 1e-9 for c in vec)


def test_vec_sum():
    assert vec_sum((2.0, 0.0, 0.0)) == 2


def test_fire_elemental_defense():
    assert MOB_DEFENSE_VEC["fire_elemental"] == (0.9, 1.0, 0.0)


def test_fire_elemental_takes_zero_fire():
    assert damage_dealt_to_mob("fire_elemental", (0.0, 4.0, 0.0)) == 0


def test_zombie_takes_full_damage():
    assert damage_dealt_to_mob("zombie", (3.0, 0.0, 0.0)) == 3


def test_knight_half_phys():
    assert damage_dealt_to_mob("knight", (4.0, 0.0, 0.0)) == 2


# ---------------------------------------------------------------------------
# T03γ: 4-slot armour state schema
# ---------------------------------------------------------------------------

def _make_env():
    """Return a freshly reset CraftaxFullEnv."""
    from glyphbench.envs.craftax.full import CraftaxFullEnv
    env = CraftaxFullEnv()
    env.reset(seed=0)
    return env


def _make_crafting_env():
    """Return an env with the player adjacent to a table AND furnace."""
    from glyphbench.envs.craftax.full import CraftaxFullEnv, TILE_TABLE, TILE_FURNACE
    env = CraftaxFullEnv()
    env.reset(seed=0)
    # Place crafting structures adjacent to the player.
    ax, ay = env._agent_x, env._agent_y
    size = env._floor_size()
    tx = ax + 1 if ax + 1 < size else ax - 1
    fx = ax - 1 if ax - 1 >= 0 else ax + 1
    env._floors[0][ay][tx] = TILE_TABLE
    env._floors[0][ay][fx] = TILE_FURNACE
    # Ensure inventory has enough materials.
    env._inventory["iron"] = 20
    env._inventory["diamond"] = 10
    return env


def test_armor_slots_exist_after_reset():
    """_armor_slots and _armor_enchants must exist with 4 keys, all 0 after reset."""
    env = _make_env()
    assert hasattr(env, "_armor_slots"), "_armor_slots missing"
    assert hasattr(env, "_armor_enchants"), "_armor_enchants missing"
    assert set(env._armor_slots.keys()) == {"helmet", "chest", "legs", "boots"}
    assert set(env._armor_enchants.keys()) == {"helmet", "chest", "legs", "boots"}
    assert all(v == 0 for v in env._armor_slots.values())
    assert all(v == 0 for v in env._armor_enchants.values())


def test_make_iron_armor_fills_helmet_first():
    """MAKE_IRON_ARMOR puts tier 1 in the helmet slot first."""
    env = _make_crafting_env()
    env._handle_make_iron_armor()
    assert env._armor_slots["helmet"] == 1
    assert env._armor_slots["chest"] == 0
    assert env._armor_slots["legs"] == 0
    assert env._armor_slots["boots"] == 0


def test_chaining_4_iron_crafts_fills_all_slots():
    """Four successive MAKE_IRON_ARMOR calls fill all four slots at tier 1."""
    env = _make_crafting_env()
    for _ in range(4):
        env._handle_make_iron_armor()
    assert all(env._armor_slots[s] == 1 for s in ("helmet", "chest", "legs", "boots"))


def test_make_iron_armor_noop_when_all_slots_filled():
    """MAKE_IRON_ARMOR is a no-op when all slots are already at tier >= 1."""
    env = _make_crafting_env()
    for s in ("helmet", "chest", "legs", "boots"):
        env._armor_slots[s] = 1
    iron_before = env._inventory["iron"]
    env._handle_make_iron_armor()
    assert env._inventory["iron"] == iron_before  # nothing consumed


def test_make_diamond_armor_upgrades_helmet_when_all_iron():
    """MAKE_DIAMOND_ARMOR upgrades the helmet (lowest tier) to tier 2."""
    env = _make_crafting_env()
    for s in ("helmet", "chest", "legs", "boots"):
        env._armor_slots[s] = 1  # all iron
    env._handle_make_diamond_armor()
    assert env._armor_slots["helmet"] == 2
    assert env._armor_slots["chest"] == 1  # unchanged


def test_action_spec_missing_wood_stone_armor():
    """MAKE_WOOD_ARMOR and MAKE_STONE_ARMOR must be absent from the action spec (T03γ)."""
    from glyphbench.envs.craftax.base import CRAFTAX_FULL_ACTION_SPEC
    assert "MAKE_WOOD_ARMOR" not in CRAFTAX_FULL_ACTION_SPEC.names
    assert "MAKE_STONE_ARMOR" not in CRAFTAX_FULL_ACTION_SPEC.names


def test_action_spec_has_41_actions():
    """Action spec has exactly 41 actions post-T03γ (43 - 2 removed = 41)."""
    from glyphbench.envs.craftax.base import CRAFTAX_FULL_ACTION_SPEC
    assert len(CRAFTAX_FULL_ACTION_SPEC.names) == 41


# ---------------------------------------------------------------------------
# T04γ: Multiplicative armour defense formula
# ---------------------------------------------------------------------------

def _no_armor():
    return {"helmet": 0, "chest": 0, "legs": 0, "boots": 0}


def _all_iron():
    return {"helmet": 1, "chest": 1, "legs": 1, "boots": 1}


def _all_enchants(element: int):
    return {"helmet": element, "chest": element, "legs": element, "boots": element}


def test_player_defense_vec_no_armor():
    """No armour → (0.0, 0.0, 0.0) defense vector."""
    assert player_defense_vec(_no_armor(), _no_armor()) == (0.0, 0.0, 0.0)


def test_player_defense_vec_4_iron_no_enchant():
    """4 iron slots, no enchants → (0.4, 0, 0)."""
    dv = player_defense_vec(_all_iron(), _no_armor())
    assert abs(dv[0] - 0.4) < 1e-9
    assert dv[1] == 0.0
    assert dv[2] == 0.0


def test_player_defense_vec_4_iron_all_fire():
    """4 iron + all fire-enchanted → (0.4, 0.8, 0.0)."""
    dv = player_defense_vec(_all_iron(), _all_enchants(1))
    assert abs(dv[0] - 0.4) < 1e-9
    assert abs(dv[1] - 0.8) < 1e-9
    assert dv[2] == 0.0


def test_damage_dealt_to_player_all_iron_phys_only():
    """All iron, no enchants, taking (10, 0, 0) → 6 (40% phys reduction)."""
    result = damage_dealt_to_player(_all_iron(), _no_armor(), (10.0, 0.0, 0.0))
    assert result == 6


def test_damage_dealt_to_player_all_iron_fire_enchant_mixed():
    """All iron + all fire enchants, taking (5, 5, 0) → 3+1 = 4."""
    # 5 phys * (1 - 0.4) = 3; 5 fire * (1 - 0.8) = 1; total = 4.
    result = damage_dealt_to_player(_all_iron(), _all_enchants(1), (5.0, 5.0, 0.0))
    assert result == 4


def test_take_damage_legacy_scalar_applies_armor():
    """_take_damage(2) with iron armor applies 40% reduction → 1 damage."""
    env = _make_env()
    for s in ("helmet", "chest", "legs", "boots"):
        env._armor_slots[s] = 1
    hp_before = env._hp
    env._take_damage(2)
    # 2 * (1 - 0.4) = 1.2 → rounds to 1.
    assert env._hp == hp_before - 1


def test_take_damage_legacy_scalar_no_armor():
    """_take_damage(3) with no armor → 3 damage (no reduction)."""
    env = _make_env()
    hp_before = env._hp
    env._take_damage(3)
    assert env._hp == hp_before - 3
