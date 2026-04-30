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


def test_action_spec_has_44_actions():
    """Action spec has exactly 44 actions post-T08γ (41 + 3 LEVEL_UP = 44)."""
    from glyphbench.envs.craftax.base import CRAFTAX_FULL_ACTION_SPEC
    assert len(CRAFTAX_FULL_ACTION_SPEC.names) == 44


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


# ---------------------------------------------------------------------------
# T05γ: Boss-floor 1.5× damage multiplier
# ---------------------------------------------------------------------------

def _make_env_on_boss_floor() -> "CraftaxFullEnv":  # type: ignore[name-defined]
    """Return an env positioned on floor 5 with a live boss mob (triggers _is_in_boss_fight)."""
    from glyphbench.envs.craftax.full import CraftaxFullEnv
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._current_floor = 5
    # Inject a boss mob on the current floor so _is_in_boss_fight() returns True.
    env._mobs.append({
        "type": "boss",
        "x": 5, "y": 5,
        "hp": 10, "max_hp": 10,
        "floor": 5,
        "is_boss": True,
        "attack_cooldown": 0,
        "facing": (1, 0),
    })
    return env


def test_take_damage_off_boss_floor_no_multiplier():
    """On floor 0 (no boss), _take_damage(4) → 4 damage (no boss multiplier)."""
    env = _make_env()  # floor 0, no boss
    hp_before = env._hp
    env._take_damage(4)
    assert env._hp == hp_before - 4


def test_take_damage_on_boss_floor_1_5x():
    """On boss floor with live boss, _take_damage(4) → 6 damage (4 × 1.5 = 6)."""
    env = _make_env_on_boss_floor()
    hp_before = env._hp
    env._take_damage(4)
    assert env._hp == hp_before - 6


# ---------------------------------------------------------------------------
# T06γ: XP state field + first-floor-entry grants
# ---------------------------------------------------------------------------

def test_xp_starts_at_zero():
    """_xp is 0 after reset."""
    env = _make_env()
    assert env._xp == 0


def test_xp_attributes_start_at_one():
    """_dex, _str, _int_attr all start at 1 after reset."""
    env = _make_env()
    assert env._dex == 1
    assert env._str == 1
    assert env._int_attr == 1


def _teleport_to_floor1_stairs(env) -> None:
    """Place the player on the stairs-down tile of floor 0 so DESCEND works."""
    pos = env._stairs_down_pos.get(0)
    if pos:
        env._agent_x, env._agent_y = pos
    else:
        # Find stairs-down manually
        grid = env._floors[0]
        from glyphbench.envs.craftax.base import TILE_STAIRS_DOWN
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if cell == TILE_STAIRS_DOWN:
                    env._agent_x, env._agent_y = x, y
                    return


def test_descend_to_floor1_grants_1_xp():
    """First descent to floor 1 grants +1 XP."""
    env = _make_env()
    assert env._xp == 0
    _teleport_to_floor1_stairs(env)
    env._handle_descend()
    assert env._xp == 1


def test_descend_to_floor1_twice_no_double_grant():
    """Re-descending to floor 1 (ascend then descend again) does NOT grant more XP."""
    env = _make_env()
    _teleport_to_floor1_stairs(env)
    env._handle_descend()
    assert env._xp == 1
    # Ascend back to surface
    env._handle_ascend()
    # Descend again
    _teleport_to_floor1_stairs(env)
    env._handle_descend()
    # Still only 1 XP
    assert env._xp == 1


# ---------------------------------------------------------------------------
# T08γ: LEVEL_UP_DEXTERITY/STRENGTH/INTELLIGENCE actions
# ---------------------------------------------------------------------------

def test_level_up_dexterity_action_exists():
    """LEVEL_UP_DEXTERITY must be in the action spec."""
    from glyphbench.envs.craftax.base import CRAFTAX_FULL_ACTION_SPEC
    assert "LEVEL_UP_DEXTERITY" in CRAFTAX_FULL_ACTION_SPEC.names
    assert "LEVEL_UP_STRENGTH" in CRAFTAX_FULL_ACTION_SPEC.names
    assert "LEVEL_UP_INTELLIGENCE" in CRAFTAX_FULL_ACTION_SPEC.names


def test_level_up_dexterity_no_xp_is_noop():
    """LEVEL_UP_DEXTERITY with _xp = 0 is a no-op (dex unchanged, no XP spent)."""
    env = _make_env()
    env._xp = 0
    env._handle_level_up_dexterity()
    assert env._dex == 1
    assert env._xp == 0


def test_level_up_dexterity_consumes_xp_and_raises_dex():
    """LEVEL_UP_DEXTERITY with _xp = 1 consumes 1 XP and raises _dex to 2."""
    env = _make_env()
    env._xp = 1
    env._handle_level_up_dexterity()
    assert env._dex == 2
    assert env._xp == 0


def test_level_up_dexterity_at_cap_is_noop():
    """LEVEL_UP_DEXTERITY when _dex = 5 is a no-op (cap enforced)."""
    env = _make_env()
    env._dex = 5
    env._xp = 3
    env._handle_level_up_dexterity()
    assert env._dex == 5
    assert env._xp == 3  # XP not consumed


def test_level_up_strength_works():
    """LEVEL_UP_STRENGTH with _xp = 1 raises _str from 1 to 2."""
    env = _make_env()
    env._xp = 1
    env._handle_level_up_strength()
    assert env._str == 2
    assert env._xp == 0


def test_level_up_intelligence_works():
    """LEVEL_UP_INTELLIGENCE with _xp = 1 raises _int_attr from 1 to 2."""
    env = _make_env()
    env._xp = 1
    env._handle_level_up_intelligence()
    assert env._int_attr == 2
    assert env._xp == 0
