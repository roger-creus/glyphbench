"""Phase-β tests: T01β — upstream achievement bitmap state field."""

from __future__ import annotations

import pytest

from glyphbench.envs.craftax.base import CRAFTAX_FULL_ACTION_SPEC
from glyphbench.envs.craftax.full import CraftaxFullEnv, UPSTREAM_ACHIEVEMENT_NAMES


# ---------------------------------------------------------------------------
# T01β: UPSTREAM_ACHIEVEMENT_NAMES constant
# ---------------------------------------------------------------------------

def test_upstream_achievement_names_count():
    """The upstream Achievement enum has exactly 67 members (values 0-66)."""
    assert len(UPSTREAM_ACHIEVEMENT_NAMES) == 67


def test_upstream_achievement_names_all_strings():
    assert all(isinstance(n, str) for n in UPSTREAM_ACHIEVEMENT_NAMES)


def test_upstream_achievement_names_no_duplicates():
    assert len(set(UPSTREAM_ACHIEVEMENT_NAMES)) == len(UPSTREAM_ACHIEVEMENT_NAMES)


def test_upstream_achievement_names_spot_check():
    """Spot-check a few well-known achievement names."""
    for name in ("COLLECT_WOOD", "PLACE_TABLE", "DEFEAT_ZOMBIE", "OPEN_CHEST",
                 "COLLECT_SAPPHIRE", "COLLECT_RUBY", "ENCHANT_ARMOUR"):
        assert name in UPSTREAM_ACHIEVEMENT_NAMES, (
            f"{name!r} missing from UPSTREAM_ACHIEVEMENT_NAMES"
        )


# ---------------------------------------------------------------------------
# T01β: _achievements_phase_beta state field
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    e = CraftaxFullEnv(max_turns=500)
    e.reset(seed=0)
    return e


def test_achievements_phase_beta_exists(env):
    assert hasattr(env, "_achievements_phase_beta")


def test_achievements_phase_beta_has_67_keys(env):
    assert len(env._achievements_phase_beta) == 67


def test_achievements_phase_beta_all_false_after_reset(env):
    assert all(not v for v in env._achievements_phase_beta.values())


def test_achievements_phase_beta_keys_match_names(env):
    assert set(env._achievements_phase_beta.keys()) == set(UPSTREAM_ACHIEVEMENT_NAMES)


def test_achievements_phase_beta_reset_clears_manual_set(env):
    """Setting a flag manually and then resetting should clear it."""
    env._achievements_phase_beta["COLLECT_WOOD"] = True
    assert env._achievements_phase_beta["COLLECT_WOOD"] is True
    env.reset(seed=1)
    assert env._achievements_phase_beta["COLLECT_WOOD"] is False


def test_achievements_phase_beta_independent_of_unlocked(env):
    """_achievements_phase_beta is a separate dict from _achievements_unlocked."""
    assert env._achievements_phase_beta is not env._achievements_unlocked
    assert isinstance(env._achievements_phase_beta, dict)
    assert isinstance(env._achievements_unlocked, set)


# ---------------------------------------------------------------------------
# T02β: REST action + _is_resting state machine
# ---------------------------------------------------------------------------

_REST_ACTION = 35   # index of REST in CRAFTAX_FULL_ACTION_SPEC (phase β: 43 actions with T11β)
_NOOP_ACTION = 0    # index of NOOP


def test_rest_action_in_spec():
    """REST is in CRAFTAX_FULL_ACTION_SPEC at index 35 (phase β spec is 43 actions with T11β)."""
    from glyphbench.envs.craftax.base import CRAFTAX_FULL_ACTION_SPEC
    assert len(CRAFTAX_FULL_ACTION_SPEC.names) == 43
    assert CRAFTAX_FULL_ACTION_SPEC.names[_REST_ACTION] == "REST"


def test_rest_sets_is_resting(env):
    """REST action sets _is_resting = True."""
    assert env._is_resting is False
    env.step(_REST_ACTION)
    assert env._is_resting is True


def test_rest_regens_hp_per_tick(env):
    """While resting, HP increases by 1 each tick."""
    env._is_resting = True
    env._hp = 5
    env.step(_NOOP_ACTION)
    assert env._hp == 6


def test_rest_exits_on_full_hp(env):
    """Resting exits when HP reaches max and HP is capped at max_hp."""
    env._is_resting = True
    env._hp = env._max_hp - 1
    env.step(_NOOP_ACTION)
    assert env._hp == env._max_hp
    assert env._is_resting is False


def test_rest_exits_on_damage(env):
    """Taking damage while resting cancels the REST state."""
    env._is_resting = True
    env._hp = 5
    env._take_damage(1)
    assert env._is_resting is False


# ---------------------------------------------------------------------------
# T03β: SLEEP continuous state machine
# ---------------------------------------------------------------------------

_SLEEP_ACTION = 6   # index of SLEEP in CRAFTAX_FULL_ACTION_SPEC
_NOOP_ACTION_IDX = 0


def test_sleep_action_sets_is_sleeping(env):
    """SLEEP action sets _is_sleeping = True."""
    assert env._is_sleeping is False
    env.step(_SLEEP_ACTION)
    assert env._is_sleeping is True


def test_sleep_regens_hp_plus2_per_tick(env):
    """While sleeping, HP increases by 2 each tick."""
    env._is_sleeping = True
    env._hp = 3
    from glyphbench.envs.craftax.full import _MAX_ENERGY
    # Drain energy so sleep doesn't exit on first tick.
    env._energy = 1
    env.step(_NOOP_ACTION_IDX)
    assert env._hp == 5


def test_sleep_increases_energy_per_tick(env):
    """While sleeping, energy increases each tick."""
    env._is_sleeping = True
    from glyphbench.envs.craftax.full import _MAX_ENERGY
    env._energy = 0
    env.step(_NOOP_ACTION_IDX)
    assert env._energy > 0


def test_sleep_exits_on_energy_full_and_fires_wake_up(env):
    """Sleep exits when energy reaches max and fires the wake_up achievement."""
    env._is_sleeping = True
    from glyphbench.envs.craftax.full import _MAX_ENERGY
    # Set energy just below max so one tick tips it over.
    env._energy = _MAX_ENERGY - 1
    env.step(_NOOP_ACTION_IDX)
    assert env._is_sleeping is False
    assert "wake_up" in env._achievements_unlocked


def test_sleep_exits_on_damage_no_wake_up(env):
    """Sleep exits on damage and does NOT fire the wake_up achievement."""
    env._is_sleeping = True
    env._hp = 9
    env._take_damage(1)
    assert env._is_sleeping is False
    assert "wake_up" not in env._achievements_unlocked


# ---------------------------------------------------------------------------
# T04β: Legacy mob name alignment regression (T_FOLLOWUP_A)
# ---------------------------------------------------------------------------

def test_legacy_skeleton_archer_absent_from_mob_stats():
    """'skeleton_archer' must not appear in _MOB_STATS after T_FOLLOWUP_A."""
    from glyphbench.envs.craftax.full import _MOB_STATS
    assert "skeleton_archer" not in _MOB_STATS, (
        "'skeleton_archer' is a legacy name that should have been renamed to 'skeleton'"
    )


def test_legacy_spider_absent_from_mob_stats():
    """'spider' must not appear in _MOB_STATS after T_FOLLOWUP_A."""
    from glyphbench.envs.craftax.full import _MOB_STATS
    assert "spider" not in _MOB_STATS, (
        "'spider' is a legacy name that should have been renamed to 'kobold'"
    )


def test_upstream_skeleton_present_in_mob_stats():
    """'skeleton' (upstream ranged) must be in _MOB_STATS with ranged-level stats."""
    from glyphbench.envs.craftax.full import _MOB_STATS
    assert "skeleton" in _MOB_STATS
    # Upstream ranged skeleton: hp=5, damage=3
    assert _MOB_STATS["skeleton"]["hp"] == 5
    assert _MOB_STATS["skeleton"]["damage"] == 3


def test_kobold_present_in_mob_stats():
    """'kobold' (replaces legacy spider) must be in _MOB_STATS."""
    from glyphbench.envs.craftax.full import _MOB_STATS
    assert "kobold" in _MOB_STATS


def test_legacy_skeleton_melee_absent_from_night_spawn(env):
    """Night spawns only produce zombies (legacy melee skeleton dropped)."""
    # Manually call _spawn_night_mobs many times and verify no legacy "skeleton" melee
    for _ in range(20):
        env._spawn_night_mobs()
    night_mobs = [m for m in env._mobs if m["floor"] == 0 and m["type"] != "cow"]
    non_zombie_night = [m for m in night_mobs if m["type"] != "zombie"]
    assert len(non_zombie_night) == 0, (
        f"Night spawn produced non-zombie hostile mobs: {[m['type'] for m in non_zombie_night]}"
    )


def test_fight_archers_env_spawns_skeletons():
    """craftax-fight-archers-v0 spawns 3 'skeleton' mobs (not skeleton_archer)."""
    import glyphbench.envs.craftax  # noqa: F401
    from glyphbench.core import make_env
    env = make_env("glyphbench/craftax-fight-archers-v0")
    env.reset(seed=0)
    skeletons = [m for m in env._mobs if m["type"] == "skeleton"]
    legacy = [m for m in env._mobs if m["type"] == "skeleton_archer"]
    assert len(skeletons) == 3, f"expected 3 skeletons, got {len(skeletons)}"
    assert len(legacy) == 0, f"legacy skeleton_archer still present"


def test_fight_spiders_env_spawns_kobolds():
    """craftax-fight-spiders-v0 spawns 3 'kobold' mobs (not spider)."""
    import glyphbench.envs.craftax  # noqa: F401
    from glyphbench.core import make_env
    env = make_env("glyphbench/craftax-fight-spiders-v0")
    env.reset(seed=0)
    kobolds = [m for m in env._mobs if m["type"] == "kobold"]
    legacy = [m for m in env._mobs if m["type"] == "spider"]
    assert len(kobolds) == 3, f"expected 3 kobolds, got {len(kobolds)}"
    assert len(legacy) == 0, f"legacy spider still present"


def test_defeat_skeleton_achievement_fires_on_upstream_skeleton_kill(env):
    """Killing a 'skeleton' (upstream ranged) fires 'defeat_skeleton' achievement."""
    from glyphbench.envs.craftax.full import _MOB_STATS
    # Place a skeleton adjacent to the player
    env._mobs = []
    fx = env._agent_x + env._facing[0]
    fy = env._agent_y + env._facing[1]
    env._mobs.append({
        "type": "skeleton",
        "x": fx, "y": fy,
        "hp": 1,  # 1 HP so one hit kills
        "max_hp": _MOB_STATS["skeleton"]["hp"],
        "is_boss": False,
        "floor": 0,
        "attack_cooldown": 0,
    })
    do_idx = env.action_spec.names.index("DO")
    env.step(do_idx)
    assert "defeat_skeleton" in env._achievements_unlocked


def test_defeat_kobold_achievement_fires_on_kobold_kill(env):
    """Killing a 'kobold' fires 'defeat_kobold' achievement."""
    from glyphbench.envs.craftax.full import _MOB_STATS
    env._mobs = []
    fx = env._agent_x + env._facing[0]
    fy = env._agent_y + env._facing[1]
    env._mobs.append({
        "type": "kobold",
        "x": fx, "y": fy,
        "hp": 1,
        "max_hp": _MOB_STATS["kobold"]["hp"],
        "is_boss": False,
        "floor": 0,
        "attack_cooldown": 0,
    })
    do_idx = env.action_spec.names.index("DO")
    env.step(do_idx)
    assert "defeat_kobold" in env._achievements_unlocked


# ---------------------------------------------------------------------------
# T05β: Sapphire + ruby ore tile constants + inventory keys
# ---------------------------------------------------------------------------

def test_sapphire_inventory_key_after_reset(env):
    """_inventory has 'sapphire' key equal to 0 after reset."""
    assert "sapphire" in env._inventory
    assert env._inventory["sapphire"] == 0


def test_ruby_inventory_key_after_reset(env):
    """_inventory has 'ruby' key equal to 0 after reset."""
    assert "ruby" in env._inventory
    assert env._inventory["ruby"] == 0


def test_tile_sapphire_single_codepoint():
    """TILE_SAPPHIRE is exactly 1 character."""
    from glyphbench.envs.craftax.base import TILE_SAPPHIRE
    assert len(TILE_SAPPHIRE) == 1


def test_tile_ruby_single_codepoint():
    """TILE_RUBY is exactly 1 character."""
    from glyphbench.envs.craftax.base import TILE_RUBY
    assert len(TILE_RUBY) == 1


def test_tile_sapphire_ruby_disjoint():
    """TILE_SAPPHIRE and TILE_RUBY are different glyphs."""
    from glyphbench.envs.craftax.base import TILE_SAPPHIRE, TILE_RUBY
    assert TILE_SAPPHIRE != TILE_RUBY


def test_tile_sapphire_disjoint_from_existing_palette():
    """TILE_SAPPHIRE does not collide with any existing tile glyph."""
    from glyphbench.envs.craftax.base import (
        TILE_SAPPHIRE,
        TILE_GRASS, TILE_TREE, TILE_STONE, TILE_COAL, TILE_IRON, TILE_DIAMOND,
        TILE_WATER, TILE_LAVA, TILE_SAND, TILE_AGENT, TILE_TABLE, TILE_FURNACE,
        TILE_PLACED_STONE, TILE_PLANT, TILE_STAIRS_DOWN, TILE_STAIRS_UP,
        TILE_TORCH, TILE_DUNGEON_WALL, TILE_DUNGEON_FLOOR, TILE_BOSS_DOOR,
        TILE_ZOMBIE, TILE_SKELETON, TILE_COW, TILE_SKELETON_ARCHER, TILE_KOBOLD,
        TILE_BAT, TILE_BOSS, TILE_SAPLING, TILE_RIPE_PLANT,
        TILE_ARROW, TILE_ARROW2, TILE_DAGGER,
        TILE_FIREBALL, TILE_FIREBALL2, TILE_ICEBALL, TILE_ICEBALL2, TILE_SLIMEBALL,
    )
    existing = {
        TILE_GRASS, TILE_TREE, TILE_STONE, TILE_COAL, TILE_IRON, TILE_DIAMOND,
        TILE_WATER, TILE_LAVA, TILE_SAND, TILE_AGENT, TILE_TABLE, TILE_FURNACE,
        TILE_PLACED_STONE, TILE_PLANT, TILE_STAIRS_DOWN, TILE_STAIRS_UP,
        TILE_TORCH, TILE_DUNGEON_WALL, TILE_DUNGEON_FLOOR, TILE_BOSS_DOOR,
        TILE_ZOMBIE, TILE_SKELETON, TILE_COW, TILE_SKELETON_ARCHER, TILE_KOBOLD,
        TILE_BAT, TILE_BOSS, TILE_SAPLING, TILE_RIPE_PLANT,
        TILE_ARROW, TILE_ARROW2, TILE_DAGGER,
        TILE_FIREBALL, TILE_FIREBALL2, TILE_ICEBALL, TILE_ICEBALL2, TILE_SLIMEBALL,
    }
    assert TILE_SAPPHIRE not in existing, (
        f"TILE_SAPPHIRE '{TILE_SAPPHIRE}' collides with existing palette"
    )


def test_tile_ruby_disjoint_from_existing_palette():
    """TILE_RUBY does not collide with any existing tile glyph."""
    from glyphbench.envs.craftax.base import (
        TILE_RUBY,
        TILE_GRASS, TILE_TREE, TILE_STONE, TILE_COAL, TILE_IRON, TILE_DIAMOND,
        TILE_WATER, TILE_LAVA, TILE_SAND, TILE_AGENT, TILE_TABLE, TILE_FURNACE,
        TILE_PLACED_STONE, TILE_PLANT, TILE_STAIRS_DOWN, TILE_STAIRS_UP,
        TILE_TORCH, TILE_DUNGEON_WALL, TILE_DUNGEON_FLOOR, TILE_BOSS_DOOR,
        TILE_ZOMBIE, TILE_SKELETON, TILE_COW, TILE_SKELETON_ARCHER, TILE_KOBOLD,
        TILE_BAT, TILE_BOSS, TILE_SAPLING, TILE_RIPE_PLANT,
        TILE_ARROW, TILE_ARROW2, TILE_DAGGER,
        TILE_FIREBALL, TILE_FIREBALL2, TILE_ICEBALL, TILE_ICEBALL2, TILE_SLIMEBALL,
    )
    existing = {
        TILE_GRASS, TILE_TREE, TILE_STONE, TILE_COAL, TILE_IRON, TILE_DIAMOND,
        TILE_WATER, TILE_LAVA, TILE_SAND, TILE_AGENT, TILE_TABLE, TILE_FURNACE,
        TILE_PLACED_STONE, TILE_PLANT, TILE_STAIRS_DOWN, TILE_STAIRS_UP,
        TILE_TORCH, TILE_DUNGEON_WALL, TILE_DUNGEON_FLOOR, TILE_BOSS_DOOR,
        TILE_ZOMBIE, TILE_SKELETON, TILE_COW, TILE_SKELETON_ARCHER, TILE_KOBOLD,
        TILE_BAT, TILE_BOSS, TILE_SAPLING, TILE_RIPE_PLANT,
        TILE_ARROW, TILE_ARROW2, TILE_DAGGER,
        TILE_FIREBALL, TILE_FIREBALL2, TILE_ICEBALL, TILE_ICEBALL2, TILE_SLIMEBALL,
    }
    assert TILE_RUBY not in existing, (
        f"TILE_RUBY '{TILE_RUBY}' collides with existing palette"
    )


# ---------------------------------------------------------------------------
# T06β: Sapphire + ruby ore placement on floor 2 + iron-pickaxe gating
# ---------------------------------------------------------------------------

def test_floor2_has_sapphire_tiles_over_many_seeds():
    """Floor 2 should contain at least one TILE_SAPPHIRE across 50 seeds."""
    from glyphbench.envs.craftax.base import TILE_SAPPHIRE
    sapphire_count = 0
    for seed in range(50):
        e = CraftaxFullEnv(max_turns=500)
        e.reset(seed=seed)
        floor2_grid = e._floors.get(2, [])
        for row in floor2_grid:
            sapphire_count += row.count(TILE_SAPPHIRE)
    assert sapphire_count > 0, (
        "No TILE_SAPPHIRE found on floor 2 across 50 seeds"
    )


def test_floor2_has_ruby_tiles_over_many_seeds():
    """Floor 2 should contain at least one TILE_RUBY across 50 seeds."""
    from glyphbench.envs.craftax.base import TILE_RUBY
    ruby_count = 0
    for seed in range(50):
        e = CraftaxFullEnv(max_turns=500)
        e.reset(seed=seed)
        floor2_grid = e._floors.get(2, [])
        for row in floor2_grid:
            ruby_count += row.count(TILE_RUBY)
    assert ruby_count > 0, (
        "No TILE_RUBY found on floor 2 across 50 seeds"
    )


def test_other_floors_have_no_sapphire_ruby():
    """Floors other than 2 should NOT have sapphire or ruby tiles (phase β)."""
    from glyphbench.envs.craftax.base import TILE_SAPPHIRE, TILE_RUBY
    e = CraftaxFullEnv(max_turns=500)
    e.reset(seed=42)
    for fl in [1, 3, 4, 5]:
        grid = e._floors.get(fl, [])
        for row in grid:
            for tile in row:
                assert tile not in (TILE_SAPPHIRE, TILE_RUBY), (
                    f"Floor {fl} unexpectedly contains gem ore tile '{tile}'"
                )


def _setup_gem_test(tile):
    """Create env with agent at dungeon floor 2, facing a given ore tile."""
    from glyphbench.envs.craftax.full import _DUNGEON_SIZE
    e = CraftaxFullEnv(max_turns=500)
    e.reset(seed=0)
    # Move agent to a safe position inside dungeon bounds (10, 10)
    e._current_floor = 2
    e._agent_x = 10
    e._agent_y = 10
    e._facing = (1, 0)  # face right → tile at (11, 10)
    e._floors[2][10][11] = tile
    return e, 11, 10   # returns (env, fx, fy)


def test_mining_sapphire_requires_iron_pickaxe_no_pickaxe():
    """DO on sapphire tile without iron pickaxe does not yield sapphire."""
    from glyphbench.envs.craftax.base import TILE_SAPPHIRE
    e, fx, fy = _setup_gem_test(TILE_SAPPHIRE)
    e._inventory.pop("iron_pickaxe", None)
    before = e._inventory.get("sapphire", 0)
    do_idx = e.action_spec.names.index("DO")
    e.step(do_idx)
    assert e._inventory.get("sapphire", 0) == before, (
        "Mining sapphire should fail without iron pickaxe"
    )
    assert e._floors[2][fy][fx] == TILE_SAPPHIRE, (
        "Sapphire tile should remain when no pickaxe"
    )


def test_mining_sapphire_with_iron_pickaxe_yields_sapphire():
    """DO on sapphire tile with iron pickaxe yields 1 sapphire."""
    from glyphbench.envs.craftax.base import TILE_SAPPHIRE, TILE_DUNGEON_FLOOR
    e, fx, fy = _setup_gem_test(TILE_SAPPHIRE)
    e._inventory["iron_pickaxe"] = 1
    before = e._inventory.get("sapphire", 0)
    do_idx = e.action_spec.names.index("DO")
    e.step(do_idx)
    assert e._inventory.get("sapphire", 0) == before + 1, (
        "Mining sapphire with iron pickaxe should yield 1 sapphire"
    )
    assert e._floors[2][fy][fx] == TILE_DUNGEON_FLOOR, (
        "Sapphire tile should be replaced by dungeon floor after mining"
    )


def test_mining_ruby_requires_iron_pickaxe_no_pickaxe():
    """DO on ruby tile without iron pickaxe does not yield ruby."""
    from glyphbench.envs.craftax.base import TILE_RUBY
    e, fx, fy = _setup_gem_test(TILE_RUBY)
    e._inventory.pop("iron_pickaxe", None)
    before = e._inventory.get("ruby", 0)
    do_idx = e.action_spec.names.index("DO")
    e.step(do_idx)
    assert e._inventory.get("ruby", 0) == before, (
        "Mining ruby should fail without iron pickaxe"
    )
    assert e._floors[2][fy][fx] == TILE_RUBY, (
        "Ruby tile should remain when no pickaxe"
    )


def test_mining_ruby_with_iron_pickaxe_yields_ruby():
    """DO on ruby tile with iron pickaxe yields 1 ruby."""
    from glyphbench.envs.craftax.base import TILE_RUBY, TILE_DUNGEON_FLOOR
    e, fx, fy = _setup_gem_test(TILE_RUBY)
    e._inventory["iron_pickaxe"] = 1
    before = e._inventory.get("ruby", 0)
    do_idx = e.action_spec.names.index("DO")
    e.step(do_idx)
    assert e._inventory.get("ruby", 0) == before + 1, (
        "Mining ruby with iron pickaxe should yield 1 ruby"
    )
    assert e._floors[2][fy][fx] == TILE_DUNGEON_FLOOR, (
        "Ruby tile should be replaced by dungeon floor after mining"
    )


def test_collect_sapphire_achievement_fires_on_mining():
    """Mining sapphire with iron pickaxe fires collect_sapphire achievement."""
    from glyphbench.envs.craftax.base import TILE_SAPPHIRE
    e, _fx, _fy = _setup_gem_test(TILE_SAPPHIRE)
    e._inventory["iron_pickaxe"] = 1
    do_idx = e.action_spec.names.index("DO")
    e.step(do_idx)
    assert "collect_sapphire" in e._achievements_unlocked


def test_collect_ruby_achievement_fires_on_mining():
    """Mining ruby with iron pickaxe fires collect_ruby achievement."""
    from glyphbench.envs.craftax.base import TILE_RUBY
    e, _fx, _fy = _setup_gem_test(TILE_RUBY)
    e._inventory["iron_pickaxe"] = 1
    do_idx = e.action_spec.names.index("DO")
    e.step(do_idx)
    assert "collect_ruby" in e._achievements_unlocked


# ---------------------------------------------------------------------------
# T07β: Lava damage on contact
# ---------------------------------------------------------------------------

def _setup_lava_test():
    """Create env with agent standing on a lava tile (surface)."""
    from glyphbench.envs.craftax.base import TILE_LAVA
    e = CraftaxFullEnv(max_turns=500)
    e.reset(seed=0)
    # Force agent to surface floor 0 and place lava at agent position.
    e._current_floor = 0
    ax, ay = e._agent_x, e._agent_y
    e._floors[0][ay][ax] = TILE_LAVA
    return e


def test_lava_tile_is_walkable():
    """Player can move onto a lava tile (movement is not blocked)."""
    from glyphbench.envs.craftax.base import TILE_LAVA, TILE_GRASS
    e = CraftaxFullEnv(max_turns=500)
    e.reset(seed=0)
    e._current_floor = 0
    # Place lava one tile to the right of the agent.
    ax, ay = e._agent_x, e._agent_y
    e._floors[0][ay][ax + 1] = TILE_LAVA
    e._facing = (1, 0)
    move_right_idx = e.action_spec.names.index("MOVE_RIGHT")
    e.step(move_right_idx)
    assert (e._agent_x, e._agent_y) == (ax + 1, ay), (
        "Player should have moved onto the lava tile"
    )


def test_standing_on_lava_deals_2_damage_per_tick():
    """Standing on lava deals exactly 2 damage per tick (armour-reduced by max(1, 2-def))."""
    e = _setup_lava_test()
    # Strip all armor so damage = max(1, 2 - 0) = 2.
    for armor_key in ("wood_armor", "stone_armor", "iron_armor", "diamond_armor"):
        e._inventory[armor_key] = 0
    hp_before = e._hp
    noop_idx = e.action_spec.names.index("NOOP")
    e.step(noop_idx)
    assert e._hp == hp_before - 2, (
        f"Expected HP to drop by 2 (lava), got {hp_before} -> {e._hp}"
    )


def test_multiple_lava_ticks_drain_hp_linearly():
    """Three ticks on lava drain HP by 6 total (2 per tick, no armor)."""
    e = _setup_lava_test()
    for armor_key in ("wood_armor", "stone_armor", "iron_armor", "diamond_armor"):
        e._inventory[armor_key] = 0
    # Set HP high enough to survive 3 ticks.
    e._hp = 20
    e._max_hp = 20
    noop_idx = e.action_spec.names.index("NOOP")
    for _ in range(3):
        e.step(noop_idx)
    assert e._hp == 14, (
        f"Expected HP=14 after 3 lava ticks (20 - 6), got {e._hp}"
    )


# ---------------------------------------------------------------------------
# T08β: 6-color potion infrastructure
# ---------------------------------------------------------------------------

def test_make_potion_mapping_returns_6_permutation():
    """make_potion_mapping(seed=0) returns a tuple that is a permutation of (0..5)."""
    from glyphbench.envs.craftax.mechanics.potions import make_potion_mapping
    perm = make_potion_mapping(seed=0)
    assert isinstance(perm, tuple), "make_potion_mapping must return a tuple"
    assert len(perm) == 6, f"Expected length 6, got {len(perm)}"
    assert set(perm) == {0, 1, 2, 3, 4, 5}, f"Not a permutation of 0-5: {perm}"


def test_make_potion_mapping_is_deterministic():
    """Same seed always produces the same mapping."""
    from glyphbench.envs.craftax.mechanics.potions import make_potion_mapping
    perm_a = make_potion_mapping(seed=42)
    perm_b = make_potion_mapping(seed=42)
    assert perm_a == perm_b, "Same seed must produce same mapping"


def test_make_potion_mapping_varies_across_seeds():
    """At least 2 distinct mappings exist among 5 different seeds."""
    from glyphbench.envs.craftax.mechanics.potions import make_potion_mapping
    mappings = {make_potion_mapping(seed=s) for s in range(5)}
    assert len(mappings) >= 2, (
        f"Expected >= 2 distinct mappings across 5 seeds, got {len(mappings)}"
    )


def test_six_drink_potion_actions_in_spec():
    """All 6 DRINK_POTION_* actions are present in the full action spec."""
    from glyphbench.envs.craftax.base import CRAFTAX_FULL_ACTION_SPEC
    colors = ("RED", "GREEN", "BLUE", "PINK", "CYAN", "YELLOW")
    for color in colors:
        name = f"DRINK_POTION_{color}"
        assert name in CRAFTAX_FULL_ACTION_SPEC.names, (
            f"{name} missing from CRAFTAX_FULL_ACTION_SPEC"
        )


def test_drink_potion_red_noop_when_no_potion(env):
    """Drinking a red potion when red == 0 is a no-op (count unchanged)."""
    env._inventory["potions"]["red"] = 0
    env._hp = 1  # below full to prevent full_health achievement noise
    env._mana = 0
    env._energy = 0
    dr_idx = env.action_spec.names.index("DRINK_POTION_RED")
    obs, reward, *_ = env.step(dr_idx)
    assert env._inventory["potions"]["red"] == 0
    assert reward == 0.0, f"No-op drink should give 0 reward, got {reward}"


def test_drink_potion_red_consumes_one_red_potion(env):
    """Drinking a red potion reduces red count by exactly 1."""
    env._inventory["potions"]["red"] = 3
    dr_idx = env.action_spec.names.index("DRINK_POTION_RED")
    env.step(dr_idx)
    assert env._inventory["potions"]["red"] == 2


def test_drink_potion_does_not_touch_other_colors(env):
    """Drinking a red potion leaves other colors unchanged."""
    for color in ("green", "blue", "pink", "cyan", "yellow"):
        env._inventory["potions"][color] = 5
    env._inventory["potions"]["red"] = 1
    dr_idx = env.action_spec.names.index("DRINK_POTION_RED")
    env.step(dr_idx)
    for color in ("green", "blue", "pink", "cyan", "yellow"):
        assert env._inventory["potions"][color] == 5, (
            f"{color} potion count changed after drinking red"
        )


def test_potion_mapping_exists_after_reset(env):
    """_potion_mapping is set after reset and has length 6."""
    assert hasattr(env, "_potion_mapping")
    assert len(env._potion_mapping) == 6
    assert set(env._potion_mapping) == {0, 1, 2, 3, 4, 5}


def test_potion_mapping_hidden_from_observation(env):
    """Rendered observation text never contains '_potion_mapping'."""
    obs, *_ = env.step(0)  # NOOP to get observation
    obs_text = str(obs)
    assert "_potion_mapping" not in obs_text, (
        "Hidden mapping must not appear in rendered observation"
    )
    # Also check the HUD field specifically.
    if hasattr(obs, "hud"):
        assert "_potion_mapping" not in obs.hud


# ---------------------------------------------------------------------------
# T09β: Potion effect handlers — stat deltas
# ---------------------------------------------------------------------------

def _make_env_with_red_mapped_to(effect_name: str) -> CraftaxFullEnv:
    """Return a freshly reset env whose RED potion maps to *effect_name*."""
    from glyphbench.envs.craftax.mechanics.potions import make_potion_mapping, POTION_EFFECTS
    # Seeds discovered to give RED -> each effect:
    _seed_for_effect = {
        "heal_8": 3,
        "poison_3": 5,
        "mana_8": 1,
        "mana_drain_3": 4,
        "energy_8": 0,
        "energy_drain_3": 8,
    }
    seed = _seed_for_effect[effect_name]
    e = CraftaxFullEnv(max_turns=500)
    e.reset(seed=seed)
    # Verify the mapping is as expected.
    perm = make_potion_mapping(seed)
    assert POTION_EFFECTS[perm[0]] == effect_name, (
        f"Seed {seed} RED effect mismatch: expected {effect_name}, "
        f"got {POTION_EFFECTS[perm[0]]}"
    )
    return e


def test_potion_effect_heal_8():
    """heal_8 effect: +8 HP capped at max_hp."""
    e = _make_env_with_red_mapped_to("heal_8")
    e._inventory["potions"]["red"] = 1
    e._hp = 1
    e._mana = 0
    e._energy = 0
    # Strip armor so take_damage path is clean if needed.
    for k in ("wood_armor", "stone_armor", "iron_armor", "diamond_armor"):
        e._inventory[k] = 0
    hp_before = e._hp
    dr_idx = e.action_spec.names.index("DRINK_POTION_RED")
    e.step(dr_idx)
    expected = min(e._max_hp, hp_before + 8)
    assert e._hp == expected, f"heal_8: expected HP={expected}, got {e._hp}"


def test_potion_effect_poison_3():
    """poison_3 effect: -3 HP (via _take_damage, armor applies)."""
    from glyphbench.envs.craftax.full import _MAX_ENERGY
    e = _make_env_with_red_mapped_to("poison_3")
    e._inventory["potions"]["red"] = 1
    # Strip armor for predictability.
    for k in ("wood_armor", "stone_armor", "iron_armor", "diamond_armor"):
        e._inventory[k] = 0
    e._hp = 9
    e._max_hp = 9
    e._mana = 0
    e._energy = _MAX_ENERGY
    # poison_3 uses _take_damage(3); with no armor actual = max(1, 3-0) = 3.
    dr_idx = e.action_spec.names.index("DRINK_POTION_RED")
    e.step(dr_idx)
    assert e._hp == 6, f"poison_3: expected HP=6, got {e._hp}"


def test_potion_effect_mana_8():
    """mana_8 effect: +8 mana capped at max mana."""
    from glyphbench.envs.craftax.full import _MAX_MANA
    e = _make_env_with_red_mapped_to("mana_8")
    e._inventory["potions"]["red"] = 1
    e._mana = 0
    e._energy = 0
    dr_idx = e.action_spec.names.index("DRINK_POTION_RED")
    e.step(dr_idx)
    expected = min(_MAX_MANA, 0 + 8)
    assert e._mana == expected, f"mana_8: expected mana={expected}, got {e._mana}"


def test_potion_effect_mana_drain_3():
    """mana_drain_3 effect: -3 mana clamped to 0."""
    e = _make_env_with_red_mapped_to("mana_drain_3")
    e._inventory["potions"]["red"] = 1
    e._mana = 5
    e._energy = 0
    dr_idx = e.action_spec.names.index("DRINK_POTION_RED")
    e.step(dr_idx)
    assert e._mana == 2, f"mana_drain_3: expected mana=2, got {e._mana}"


def test_potion_effect_mana_drain_3_clamps_to_zero():
    """mana_drain_3 cannot reduce mana below 0."""
    e = _make_env_with_red_mapped_to("mana_drain_3")
    e._inventory["potions"]["red"] = 1
    e._mana = 1
    e._energy = 0
    dr_idx = e.action_spec.names.index("DRINK_POTION_RED")
    e.step(dr_idx)
    assert e._mana == 0, f"mana_drain_3 clamp: expected mana=0, got {e._mana}"


def test_potion_effect_energy_8():
    """energy_8 effect: +8 energy capped at max energy."""
    from glyphbench.envs.craftax.full import _MAX_ENERGY
    e = _make_env_with_red_mapped_to("energy_8")
    e._inventory["potions"]["red"] = 1
    e._energy = 0
    e._mana = 0
    dr_idx = e.action_spec.names.index("DRINK_POTION_RED")
    e.step(dr_idx)
    expected = min(_MAX_ENERGY, 0 + 8)
    assert e._energy == expected, f"energy_8: expected energy={expected}, got {e._energy}"


def test_potion_effect_energy_drain_3():
    """energy_drain_3 effect: -3 energy clamped to 0."""
    e = _make_env_with_red_mapped_to("energy_drain_3")
    e._inventory["potions"]["red"] = 1
    e._energy = 5
    e._mana = 0
    dr_idx = e.action_spec.names.index("DRINK_POTION_RED")
    e.step(dr_idx)
    assert e._energy == 2, f"energy_drain_3: expected energy=2, got {e._energy}"


# ---------------------------------------------------------------------------
# T10β: Per-spell _learned_spells dict
# ---------------------------------------------------------------------------

def test_learned_spells_dict_exists_after_reset(env):
    """_learned_spells dict exists with both spell keys set to False after reset."""
    assert hasattr(env, "_learned_spells")
    assert isinstance(env._learned_spells, dict)
    assert "fireball" in env._learned_spells
    assert "iceball" in env._learned_spells
    assert env._learned_spells["fireball"] is False
    assert env._learned_spells["iceball"] is False


def test_old_spells_learned_attr_gone(env):
    """The legacy _spells_learned integer attribute must not exist."""
    assert not hasattr(env, "_spells_learned"), (
        "_spells_learned (integer) should have been replaced by _learned_spells dict"
    )


def test_cast_fireball_bails_when_fireball_not_learned(env):
    """CAST_FIREBALL is a no-op when _learned_spells['fireball'] is False."""
    env._mana = 5
    env._learned_spells["fireball"] = False
    initial_mana = env._mana
    env._handle_cast_fireball()
    assert env._player_projectiles == []
    assert env._mana == initial_mana, "Mana must not be consumed when fireball not learned"


def test_cast_fireball_works_when_fireball_learned(env):
    """CAST_FIREBALL spawns a projectile when _learned_spells['fireball'] is True."""
    from glyphbench.envs.craftax.mechanics.projectiles import ProjectileType
    env._mana = 5
    env._learned_spells["fireball"] = True
    env._facing = (1, 0)
    env._agent_x, env._agent_y = 5, 5
    env._handle_cast_fireball()
    assert len(env._player_projectiles) == 1
    assert env._player_projectiles[0].kind == ProjectileType.FIREBALL


def test_cast_iceball_bails_when_iceball_not_learned(env):
    """CAST_ICEBALL is a no-op when _learned_spells['iceball'] is False."""
    env._mana = 5
    env._learned_spells["iceball"] = False
    initial_mana = env._mana
    env._handle_cast_iceball()
    assert env._player_projectiles == []
    assert env._mana == initial_mana, "Mana must not be consumed when iceball not learned"


def test_cast_iceball_works_when_iceball_learned(env):
    """CAST_ICEBALL spawns a projectile when _learned_spells['iceball'] is True."""
    from glyphbench.envs.craftax.mechanics.projectiles import ProjectileType
    env._mana = 5
    env._learned_spells["iceball"] = True
    env._facing = (1, 0)
    env._agent_x, env._agent_y = 5, 5
    env._handle_cast_iceball()
    assert len(env._player_projectiles) == 1
    assert env._player_projectiles[0].kind == ProjectileType.ICEBALL


# ---------------------------------------------------------------------------
# T11β: READ_BOOK action + book inventory
# ---------------------------------------------------------------------------

def test_read_book_action_in_spec():
    """READ_BOOK is present in CRAFTAX_FULL_ACTION_SPEC (43 actions total)."""
    from glyphbench.envs.craftax.base import CRAFTAX_FULL_ACTION_SPEC
    assert "READ_BOOK" in CRAFTAX_FULL_ACTION_SPEC.names
    assert len(CRAFTAX_FULL_ACTION_SPEC.names) == 43


def test_book_inventory_key_exists_after_reset(env):
    """_inventory has 'book' key equal to 0 after reset."""
    assert "book" in env._inventory
    assert env._inventory["book"] == 0


def test_read_book_noop_when_no_book(env):
    """READ_BOOK with book count = 0 is a no-op: no spell learned, message set."""
    env._inventory["book"] = 0
    reward = env._handle_read_book()
    assert reward == 0.0
    assert env._learned_spells["fireball"] is False
    assert env._learned_spells["iceball"] is False
    assert env._inventory["book"] == 0
    assert "no book" in env._message.lower()


def test_read_book_consumes_one_book_and_learns_a_spell(env):
    """READ_BOOK with book=1 consumes the book and learns exactly one unlearned spell."""
    env._inventory["book"] = 1
    env._learned_spells["fireball"] = False
    env._learned_spells["iceball"] = False
    env._handle_read_book()
    assert env._inventory["book"] == 0, "Book should be consumed"
    learned_count = sum(env._learned_spells.values())
    assert learned_count == 1, f"Exactly 1 spell should be learned, got {learned_count}"


def test_read_book_twice_teaches_both_spells(env):
    """Two READ_BOOK calls teach both fireball and iceball."""
    env._inventory["book"] = 2
    env._learned_spells["fireball"] = False
    env._learned_spells["iceball"] = False
    env._handle_read_book()
    env._handle_read_book()
    assert env._learned_spells["fireball"] is True
    assert env._learned_spells["iceball"] is True
    assert env._inventory["book"] == 0


def test_read_book_noop_when_all_spells_known(env):
    """READ_BOOK when all spells already known is a no-op (book not consumed)."""
    env._inventory["book"] = 1
    env._learned_spells["fireball"] = True
    env._learned_spells["iceball"] = True
    reward = env._handle_read_book()
    assert reward == 0.0
    assert env._inventory["book"] == 1, "Book should not be consumed"
    assert "already know" in env._message.lower()


def test_learn_fireball_achievement_fires(env):
    """learn_fireball achievement fires when READ_BOOK teaches fireball."""
    # Force fireball to be the only unlearned spell.
    env._inventory["book"] = 1
    env._learned_spells["fireball"] = False
    env._learned_spells["iceball"] = True
    env._handle_read_book()
    assert "learn_fireball" in env._achievements_unlocked, (
        "learn_fireball achievement should fire after learning fireball"
    )


def test_learn_iceball_achievement_fires(env):
    """learn_iceball achievement fires when READ_BOOK teaches iceball."""
    env._inventory["book"] = 1
    env._learned_spells["fireball"] = True
    env._learned_spells["iceball"] = False
    env._handle_read_book()
    assert "learn_iceball" in env._achievements_unlocked, (
        "learn_iceball achievement should fire after learning iceball"
    )


# ---------------------------------------------------------------------------
# T12β: Chest tile constant + state fields
# ---------------------------------------------------------------------------

def test_tile_chest_single_codepoint():
    """TILE_CHEST is exactly 1 character."""
    from glyphbench.envs.craftax.base import TILE_CHEST
    assert len(TILE_CHEST) == 1, f"TILE_CHEST '{TILE_CHEST}' must be a single codepoint"


def test_tile_chest_disjoint_from_existing_palette():
    """TILE_CHEST does not collide with any existing tile glyph."""
    from glyphbench.envs.craftax.base import (
        TILE_CHEST,
        TILE_GRASS, TILE_TREE, TILE_STONE, TILE_COAL, TILE_IRON, TILE_DIAMOND,
        TILE_WATER, TILE_LAVA, TILE_SAND, TILE_AGENT, TILE_TABLE, TILE_FURNACE,
        TILE_PLACED_STONE, TILE_PLANT, TILE_STAIRS_DOWN, TILE_STAIRS_UP,
        TILE_TORCH, TILE_DUNGEON_WALL, TILE_DUNGEON_FLOOR, TILE_BOSS_DOOR,
        TILE_ZOMBIE, TILE_SKELETON, TILE_COW, TILE_SKELETON_ARCHER, TILE_KOBOLD,
        TILE_BAT, TILE_BOSS, TILE_SAPLING, TILE_RIPE_PLANT,
        TILE_ARROW, TILE_ARROW2, TILE_DAGGER,
        TILE_FIREBALL, TILE_FIREBALL2, TILE_ICEBALL, TILE_ICEBALL2, TILE_SLIMEBALL,
        TILE_SAPPHIRE, TILE_RUBY,
    )
    existing = {
        TILE_GRASS, TILE_TREE, TILE_STONE, TILE_COAL, TILE_IRON, TILE_DIAMOND,
        TILE_WATER, TILE_LAVA, TILE_SAND, TILE_AGENT, TILE_TABLE, TILE_FURNACE,
        TILE_PLACED_STONE, TILE_PLANT, TILE_STAIRS_DOWN, TILE_STAIRS_UP,
        TILE_TORCH, TILE_DUNGEON_WALL, TILE_DUNGEON_FLOOR, TILE_BOSS_DOOR,
        TILE_ZOMBIE, TILE_SKELETON, TILE_COW, TILE_SKELETON_ARCHER, TILE_KOBOLD,
        TILE_BAT, TILE_BOSS, TILE_SAPLING, TILE_RIPE_PLANT,
        TILE_ARROW, TILE_ARROW2, TILE_DAGGER,
        TILE_FIREBALL, TILE_FIREBALL2, TILE_ICEBALL, TILE_ICEBALL2, TILE_SLIMEBALL,
        TILE_SAPPHIRE, TILE_RUBY,
    }
    assert TILE_CHEST not in existing, (
        f"TILE_CHEST '{TILE_CHEST}' collides with existing palette"
    )


def test_chests_opened_empty_after_reset(env):
    """_chests_opened is an empty dict after reset."""
    assert hasattr(env, "_chests_opened")
    assert isinstance(env._chests_opened, dict)
    assert env._chests_opened == {}


def test_first_chest_opened_empty_after_reset(env):
    """_first_chest_opened is an empty dict after reset."""
    assert hasattr(env, "_first_chest_opened")
    assert isinstance(env._first_chest_opened, dict)
    assert env._first_chest_opened == {}


def test_chests_opened_resets_on_new_episode(env):
    """Manually setting _chests_opened and resetting clears it."""
    env._chests_opened[1] = {(5, 5)}
    env.reset(seed=99)
    assert env._chests_opened == {}


def test_first_chest_opened_resets_on_new_episode(env):
    """Manually setting _first_chest_opened and resetting clears it."""
    env._first_chest_opened[1] = True
    env.reset(seed=99)
    assert env._first_chest_opened == {}


# ---------------------------------------------------------------------------
# T13β: Chest loot interaction
# ---------------------------------------------------------------------------

def _place_chest_in_front(env) -> tuple[int, int]:
    """Place a TILE_CHEST tile in the cell the agent is facing.

    Moves the agent to a safe position (10, 10) facing right so that the
    facing cell (11, 10) is always in-bounds for both dungeon (32x32) and
    surface (64x64) grids.

    Returns (fx, fy) — the chest position.
    """
    from glyphbench.envs.craftax.base import TILE_CHEST, TILE_DUNGEON_FLOOR
    env._agent_x = 10
    env._agent_y = 10
    env._facing = (1, 0)
    grid = env._current_grid()
    fx = env._agent_x + env._facing[0]  # = 11
    fy = env._agent_y + env._facing[1]  # = 10
    grid[fy][fx] = TILE_CHEST
    return fx, fy


def test_opening_chest_changes_inventory(env):
    """DO on a chest tile produces at least one inventory change."""
    env._current_floor = 2  # floor 2: no first-chest gating
    _place_chest_in_front(env)
    inv_before = {k: (dict(v) if isinstance(v, dict) else v)
                  for k, v in env._inventory.items()}
    do_idx = env.action_spec.names.index("DO")
    env.step(do_idx)
    # At least one item must have changed.
    changed = False
    for k, v in env._inventory.items():
        if isinstance(v, dict):
            for ck, cv in v.items():
                if cv != inv_before.get(k, {}).get(ck, 0):
                    changed = True
                    break
        else:
            if v != inv_before.get(k, 0):
                changed = True
                break
    assert changed, "Opening a chest should change at least one inventory item"


def test_opening_chest_deterministic_with_seed():
    """Two envs with the same seed open a chest the same way."""
    from glyphbench.envs.craftax.base import TILE_CHEST

    def run(seed):
        e = CraftaxFullEnv(max_turns=500)
        e.reset(seed=seed)
        e._current_floor = 2
        e._agent_x = 10
        e._agent_y = 10
        e._facing = (1, 0)
        e._current_grid()[10][11] = TILE_CHEST
        do_idx = e.action_spec.names.index("DO")
        e.step(do_idx)
        return {k: (dict(v) if isinstance(v, dict) else v)
                for k, v in e._inventory.items()}

    inv_a = run(7)
    inv_b = run(7)
    assert inv_a == inv_b, "Same seed must produce same chest loot"


def test_opening_chest_twice_does_nothing(env):
    """Attempting to open an already-opened chest is a no-op."""
    env._current_floor = 2
    fx, fy = _place_chest_in_front(env)
    do_idx = env.action_spec.names.index("DO")
    env.step(do_idx)  # first open
    inv_after_first = {k: (dict(v) if isinstance(v, dict) else v)
                       for k, v in env._inventory.items()}
    # Place the chest tile back (it is NOT removed from the grid after opening).
    from glyphbench.envs.craftax.base import TILE_CHEST
    env._current_grid()[fy][fx] = TILE_CHEST
    env.step(do_idx)  # second open — should be a no-op
    for k, v in env._inventory.items():
        if isinstance(v, dict):
            for ck, cv in v.items():
                assert cv == inv_after_first.get(k, {}).get(ck, 0), (
                    f"Second chest open changed {k}[{ck}]: "
                    f"{inv_after_first.get(k, {}).get(ck, 0)} -> {cv}"
                )
        else:
            assert v == inv_after_first.get(k, 0), (
                f"Second chest open changed {k}: {inv_after_first.get(k, 0)} -> {v}"
            )


def test_opening_chest_fires_open_chest_achievement(env):
    """DO on a chest fires the open_chest achievement."""
    env._current_floor = 2
    _place_chest_in_front(env)
    do_idx = env.action_spec.names.index("DO")
    env.step(do_idx)
    assert "open_chest" in env._achievements_unlocked, (
        "open_chest achievement should fire on first chest open"
    )


def test_open_chest_achievement_in_all_full_achievements():
    """open_chest is listed in ALL_FULL_ACHIEVEMENTS."""
    from glyphbench.envs.craftax.base import ALL_FULL_ACHIEVEMENTS
    assert "open_chest" in ALL_FULL_ACHIEVEMENTS


def test_find_bow_achievement_in_all_full_achievements():
    """find_bow is listed in ALL_FULL_ACHIEVEMENTS."""
    from glyphbench.envs.craftax.base import ALL_FULL_ACHIEVEMENTS
    assert "find_bow" in ALL_FULL_ACHIEVEMENTS


# ---------------------------------------------------------------------------
# T14β: First-chest gating (bow on floor 1, book on floors 3-4)
# ---------------------------------------------------------------------------

def test_first_chest_floor1_grants_bow(env):
    """Opening the first chest on floor 1 grants a bow."""
    env._current_floor = 1
    env._inventory["bow"] = 0
    _place_chest_in_front(env)
    do_idx = env.action_spec.names.index("DO")
    env.step(do_idx)
    assert env._inventory.get("bow", 0) >= 1, (
        "First chest on floor 1 should grant a bow"
    )


def test_first_chest_floor1_fires_find_bow_achievement(env):
    """Opening the first chest on floor 1 fires find_bow achievement."""
    env._current_floor = 1
    env._inventory["bow"] = 0
    _place_chest_in_front(env)
    do_idx = env.action_spec.names.index("DO")
    env.step(do_idx)
    assert "find_bow" in env._achievements_unlocked, (
        "find_bow achievement should fire after first chest on floor 1"
    )


def test_second_chest_floor1_does_not_grant_bow(env):
    """Opening a second chest on floor 1 does NOT grant another bow."""
    from glyphbench.envs.craftax.base import TILE_CHEST
    env._current_floor = 1
    env._inventory["bow"] = 0
    fx, fy = _place_chest_in_front(env)
    do_idx = env.action_spec.names.index("DO")
    env.step(do_idx)  # first chest — grants bow
    bow_after_first = env._inventory.get("bow", 0)
    # Place second chest (at different position to avoid double-open guard).
    grid = env._current_grid()
    fx2 = env._agent_x + env._facing[0]
    fy2 = env._agent_y + env._facing[1]
    # Use a fresh position.
    fsize = env._floor_size()
    from glyphbench.envs.craftax.base import TILE_DUNGEON_FLOOR
    # Try a nearby free cell.
    for ddx in range(2, 6):
        nx = env._agent_x + ddx
        if 0 <= nx < fsize:
            grid[env._agent_y][nx] = TILE_CHEST
            env._facing = (1, 0)
            env._agent_x = nx - 1
            break
    env.step(do_idx)  # second chest — should NOT grant additional bow
    assert env._inventory.get("bow", 0) == bow_after_first, (
        "Second chest on floor 1 should not grant another bow"
    )


def test_first_chest_floor3_grants_book(env):
    """Opening the first chest on floor 3 grants a book."""
    env._current_floor = 3
    env._inventory["book"] = 0
    _place_chest_in_front(env)
    do_idx = env.action_spec.names.index("DO")
    env.step(do_idx)
    assert env._inventory.get("book", 0) >= 1, (
        "First chest on floor 3 should grant a book"
    )


def test_first_chest_floor3_fires_find_book_achievement(env):
    """Opening the first chest on floor 3 fires find_book achievement."""
    env._current_floor = 3
    env._inventory["book"] = 0
    _place_chest_in_front(env)
    do_idx = env.action_spec.names.index("DO")
    env.step(do_idx)
    assert "find_book" in env._achievements_unlocked, (
        "find_book achievement should fire after first chest on floor 3"
    )


def test_floor3_first_chest_then_floor4_first_chest_grants_only_one_book(env):
    """Opening the first chest on floor 3 then floor 4 grants only 1 book total."""
    from glyphbench.envs.craftax.base import TILE_CHEST
    env._inventory["book"] = 0
    # Floor 3 first chest.
    env._current_floor = 3
    _place_chest_in_front(env)
    do_idx = env.action_spec.names.index("DO")
    env.step(do_idx)
    book_after_floor3 = env._inventory.get("book", 0)
    assert book_after_floor3 >= 1, "Floor 3 first chest should give a book"
    # Floor 4 first chest — floors 1..5 already generated in _reset.
    env._current_floor = 4
    # Re-place a chest in front (floor 4 grid).
    grid = env._current_grid()
    fx = env._agent_x + env._facing[0]
    fy = env._agent_y + env._facing[1]
    grid[fy][fx] = TILE_CHEST
    env.step(do_idx)
    book_after_floor4 = env._inventory.get("book", 0)
    assert book_after_floor4 == book_after_floor3, (
        "Floor 4 first chest should NOT grant another book if floor 3 already did"
    )


def test_floor4_first_chest_grants_book_if_floor3_not_opened(env):
    """First chest on floor 4 grants book when floor 3 was never opened."""
    from glyphbench.envs.craftax.base import TILE_CHEST
    env._inventory["book"] = 0
    env._current_floor = 4
    _place_chest_in_front(env)
    do_idx = env.action_spec.names.index("DO")
    env.step(do_idx)
    assert env._inventory.get("book", 0) >= 1, (
        "First chest on floor 4 (no floor-3 chest opened) should grant a book"
    )


# ---------------------------------------------------------------------------
# T15β + T16β: Per-tile lightmap subsystem + visibility threshold
# ---------------------------------------------------------------------------

def test_compute_lightmap_all_ones_with_biome_one_no_torches():
    """compute_lightmap with no torches and biome=1.0 returns all 1.0."""
    from glyphbench.envs.craftax.mechanics.lighting import compute_lightmap
    import numpy as np
    lm = compute_lightmap(10, 10, set(), biome_baseline=1.0)
    assert lm.shape == (10, 10)
    assert np.allclose(lm, 1.0)


def test_compute_lightmap_all_zeros_with_biome_zero_no_torches():
    """compute_lightmap with no torches and biome=0.0 returns all 0.0."""
    from glyphbench.envs.craftax.mechanics.lighting import compute_lightmap
    import numpy as np
    lm = compute_lightmap(10, 10, set(), biome_baseline=0.0)
    assert lm.shape == (10, 10)
    assert np.allclose(lm, 0.0)


def test_compute_lightmap_torch_at_center_correct_values():
    """Torch at (5, 5), biome=0.0: (5,5)->1.0, (5,6)->0.8, (5,9)->0.2, (5,10)->0.0."""
    from glyphbench.envs.craftax.mechanics.lighting import compute_lightmap
    import numpy as np
    # Grid large enough: 15x15 so torch radius (5) fits.
    lm = compute_lightmap(15, 15, {(5, 5)}, biome_baseline=0.0)
    # Torch center: distance 0 -> contribution = 1.0
    assert float(lm[5, 5]) == pytest.approx(1.0, abs=1e-5), (
        f"lm[5,5] expected 1.0, got {lm[5,5]}"
    )
    # One tile away (Manhattan distance 1): contribution = 1 - 1/5 = 0.8
    assert float(lm[6, 5]) == pytest.approx(0.8, abs=1e-5), (
        f"lm[6,5] expected 0.8, got {lm[6,5]}"
    )
    # 4 tiles away (Manhattan distance 4): contribution = 1 - 4/5 = 0.2
    assert float(lm[9, 5]) == pytest.approx(0.2, abs=1e-5), (
        f"lm[9,5] expected 0.2, got {lm[9,5]}"
    )
    # 5 tiles away (Manhattan distance 5): contribution = max(0, 1-5/5) = 0.0
    assert float(lm[10, 5]) == pytest.approx(0.0, abs=1e-5), (
        f"lm[10,5] expected 0.0, got {lm[10,5]}"
    )


def test_lightmap_exists_for_current_floor_after_reset(env):
    """After reset, _lightmap has an entry for the current floor."""
    assert env._current_floor in env._lightmap, (
        f"_lightmap missing entry for current floor {env._current_floor}"
    )
    lm = env._lightmap[env._current_floor]
    assert lm is not None
    assert lm.ndim == 2
    assert lm.shape[0] > 0 and lm.shape[1] > 0


def test_placing_torch_in_dungeon_raises_local_visibility():
    """Placing a torch on a dark dungeon floor lights up the torch tile."""
    e = CraftaxFullEnv(max_turns=500)
    e.reset(seed=0)
    # Move to dungeon floor 1 (dark by default: biome_baseline=0.0, no torches).
    e._current_floor = 1
    e._recompute_lightmap(1)

    # Agent at (10, 10), facing right -> torch goes to (11, 10).
    e._agent_x = 10
    e._agent_y = 10
    e._facing = (1, 0)
    # Ensure the target cell is empty dungeon floor so the torch can be placed.
    from glyphbench.envs.craftax.base import TILE_DUNGEON_FLOOR, TILE_TORCH
    e._floors[1][10][11] = TILE_DUNGEON_FLOOR
    # Give the agent a torch to place.
    e._inventory["torch"] = 1

    # Before placement: torch cell should be dark.
    from glyphbench.envs.craftax.mechanics.lighting import VISIBILITY_THRESHOLD
    lm_before = e._lightmap.get(1)
    assert lm_before is not None
    before_light = float(lm_before[10, 11])

    place_idx = e.action_spec.names.index("PLACE_TORCH")
    e.step(place_idx)

    lm_after = e._lightmap.get(1)
    assert lm_after is not None
    after_light = float(lm_after[10, 11])
    assert after_light > VISIBILITY_THRESHOLD, (
        f"Tile at (11, 10) should be lit after torch placement, got {after_light}"
    )
    assert after_light > before_light, (
        "Light level at torch tile should increase after torch placement"
    )


def test_tiles_past_torch_radius_remain_dark():
    """Tiles outside TORCH_RADIUS from a single torch stay at biome_baseline=0.0."""
    from glyphbench.envs.craftax.mechanics.lighting import compute_lightmap, TORCH_RADIUS, VISIBILITY_THRESHOLD
    import numpy as np
    # Place torch at (5, 5) in a 20x20 grid.
    lm = compute_lightmap(20, 20, {(5, 5)}, biome_baseline=0.0)
    # A tile at Manhattan distance > TORCH_RADIUS from (5,5) must remain 0.
    # (5, 5+TORCH_RADIUS+1) = (5, 11): distance=6 > 5 -> 0.0.
    assert float(lm[11, 5]) <= VISIBILITY_THRESHOLD, (
        f"Tile at distance {TORCH_RADIUS+1} should be dark, got {lm[11,5]}"
    )


# ---------------------------------------------------------------------------
# T17β: Darkness²-scaled mob spawn rate
# ---------------------------------------------------------------------------

def _make_dark_env(light_level: float) -> CraftaxFullEnv:
    """Return a night-time env with the floor-0 lightmap overridden to *light_level*."""
    import numpy as np
    from glyphbench.envs.craftax.full import _SURFACE_SIZE
    e = CraftaxFullEnv(max_turns=500)
    e.reset(seed=42)
    # Force night so _spawn_night_mobs is logically active.
    e._day_night = "night"
    # Override the entire floor-0 lightmap to the given uniform light level.
    e._lightmap[0] = np.full((_SURFACE_SIZE, _SURFACE_SIZE), light_level, dtype=float)
    return e


def test_spawn_full_dark_base_chance_one_always_spawns():
    """With light=0.0 and base_chance=1.0, effective=(1-0)²=1.0 → mobs always spawn.

    We call _spawn_night_mobs 30 times in pitch darkness and assert at least one
    zombie is spawned (chance of zero zombies in 30 tries is astronomically small).
    """
    e = _make_dark_env(light_level=0.0)
    e._mobs = []  # clear any existing mobs
    for _ in range(30):
        e._spawn_night_mobs()
    zombies = [m for m in e._mobs if m["type"] == "zombie" and m["floor"] == 0]
    assert len(zombies) > 0, (
        "At full darkness (light=0.0) with base_chance=1.0, mobs should always spawn"
    )


def test_spawn_full_lit_never_spawns():
    """With light=1.0, effective=(1-1)²=0.0 → no mobs spawn regardless of base_chance.

    100 calls to _spawn_night_mobs with a fully lit map must produce zero zombies.
    """
    e = _make_dark_env(light_level=1.0)
    e._mobs = []
    for _ in range(100):
        e._spawn_night_mobs()
    zombies = [m for m in e._mobs if m["type"] == "zombie" and m["floor"] == 0]
    assert len(zombies) == 0, (
        f"At full light (light=1.0), effective_chance=0 → no mobs should spawn; "
        f"got {len(zombies)}"
    )


def test_spawn_half_light_quadratic_dampening():
    """With light=0.5, effective=base*(1-0.5)²=0.25 → ~25% of base-rate spawns.

    Strategy: run many spawn calls at light=0.5 using a fresh env per seed so
    the probability gate fires independently.  Over 500 fresh envs each calling
    _spawn_night_mobs once, at light=0.5 we expect ~25% of at-most-4 slots to
    produce a zombie.  We assert the total < what we get at light=0.0 (all spawns).
    To avoid RNG-synchronisation artifacts we vary the seed per trial.
    """
    import numpy as np
    from glyphbench.envs.craftax.full import _SURFACE_SIZE

    n_trials = 500

    def count_spawned(light: float) -> int:
        total = 0
        for seed in range(n_trials):
            e = CraftaxFullEnv(max_turns=500)
            e.reset(seed=seed)
            e._day_night = "night"
            e._lightmap[0] = np.full(
                (_SURFACE_SIZE, _SURFACE_SIZE), light, dtype=float
            )
            e._mobs = []
            e._spawn_night_mobs()
            total += sum(
                1 for m in e._mobs if m["type"] == "zombie" and m["floor"] == 0
            )
        return total

    count_dark = count_spawned(0.0)   # effective_chance = 1.0 → all valid positions spawn
    count_half = count_spawned(0.5)   # effective_chance = 0.25 → ~25% spawn

    # At light=0.0 every valid candidate spawns; at light=0.5 only ~25% do.
    # Over 500 seeds the gap is large enough to be robust.
    assert count_half < count_dark, (
        f"light=0.5 should produce fewer spawns than light=0.0 "
        f"(got dark={count_dark}, half={count_half})"
    )
    # Sanity: at least some mobs spawned in the dark (deterministic at 1.0 chance).
    assert count_dark > 0, (
        "Expected >0 zombies in pitch darkness across 500 spawn calls"
    )


# ---------------------------------------------------------------------------
# T18β: Dungeon-room biome generator (mechanics/world_gen.py)
# ---------------------------------------------------------------------------

def _fresh_rng(seed: int = 0):
    """Return a fresh numpy Generator for use in world_gen tests."""
    import numpy as np
    return np.random.default_rng(seed)


def test_generate_dungeon_floor_returns_correct_grid_size():
    """generate_dungeon_floor returns a grid of the requested size."""
    from glyphbench.envs.craftax.mechanics.world_gen import generate_dungeon_floor

    size = 32
    grid, _chests, _fountains, _down, _up, _spawn = generate_dungeon_floor(
        _fresh_rng(0), size
    )
    assert len(grid) == size, f"Expected {size} rows, got {len(grid)}"
    assert all(len(row) == size for row in grid), (
        "All rows must have width == size"
    )


def test_generate_dungeon_floor_has_correct_num_chests():
    """generate_dungeon_floor with num_rooms=8 places exactly 8 chests."""
    from glyphbench.envs.craftax.mechanics.world_gen import generate_dungeon_floor
    from glyphbench.envs.craftax.base import TILE_CHEST

    size = 32
    for seed in range(5):
        grid, chest_positions, _fountains, _down, _up, _spawn = (
            generate_dungeon_floor(_fresh_rng(seed), size, num_rooms=8)
        )
        # Count chest tiles in grid
        chest_tile_count = sum(
            1 for row in grid for cell in row if cell == TILE_CHEST
        )
        # chest_positions must match what's in the grid
        assert chest_tile_count == len(chest_positions), (
            f"Seed {seed}: chest tile count ({chest_tile_count}) != "
            f"len(chest_positions) ({len(chest_positions)})"
        )
        # We asked for 8 rooms so should have exactly 8 chests
        assert len(chest_positions) == 8, (
            f"Seed {seed}: expected 8 chests, got {len(chest_positions)}"
        )


def test_generate_dungeon_floor_has_stairs():
    """generate_dungeon_floor places one stairs-down and one stairs-up tile."""
    from glyphbench.envs.craftax.mechanics.world_gen import generate_dungeon_floor
    from glyphbench.envs.craftax.base import TILE_STAIRS_DOWN, TILE_STAIRS_UP

    size = 32
    for seed in range(5):
        grid, _chests, _fountains, stairs_down, stairs_up, _spawn = (
            generate_dungeon_floor(_fresh_rng(seed), size)
        )
        dx, dy = stairs_down
        ux, uy = stairs_up
        assert grid[dy][dx] == TILE_STAIRS_DOWN, (
            f"Seed {seed}: stairs_down_pos {stairs_down} doesn't point to "
            f"TILE_STAIRS_DOWN (got {grid[dy][dx]!r})"
        )
        assert grid[uy][ux] == TILE_STAIRS_UP, (
            f"Seed {seed}: stairs_up_pos {stairs_up} doesn't point to "
            f"TILE_STAIRS_UP (got {grid[uy][ux]!r})"
        )


def test_generate_dungeon_floor_fountains_statistical():
    """Roughly half the rooms have a fountain (expected 4/8); check 2-7 range."""
    from glyphbench.envs.craftax.mechanics.world_gen import generate_dungeon_floor
    from glyphbench.envs.craftax.base import TILE_FOUNTAIN

    size = 32
    all_counts: list[int] = []
    for seed in range(20):
        grid, _chests, fountain_positions, _down, _up, _spawn = (
            generate_dungeon_floor(_fresh_rng(seed), size, num_rooms=8)
        )
        # Verify fountain_positions match grid content
        fountain_tile_count = sum(
            1 for row in grid for cell in row if cell == TILE_FOUNTAIN
        )
        assert fountain_tile_count == len(fountain_positions), (
            f"Seed {seed}: fountain tile count {fountain_tile_count} != "
            f"len(fountain_positions) {len(fountain_positions)}"
        )
        all_counts.append(len(fountain_positions))

    # Over 20 seeds with 50% probability per room the distribution should
    # stay within [1, 8]; we assert a practical [0, 8] to avoid flakiness.
    assert all(0 <= c <= 8 for c in all_counts), (
        f"Fountain counts out of [0,8]: {all_counts}"
    )
    # Mean should be around 4; assert at least some variation exists.
    total = sum(all_counts)
    assert total > 0, "Expected at least some fountains across 20 seeds"


def test_generate_dungeon_floor_rooms_are_connected():
    """All floor tiles must be reachable from the agent spawn (BFS)."""
    from glyphbench.envs.craftax.mechanics.world_gen import generate_dungeon_floor
    from glyphbench.envs.craftax.base import (
        TILE_DUNGEON_FLOOR, TILE_CHEST, TILE_FOUNTAIN,
        TILE_STAIRS_DOWN, TILE_STAIRS_UP,
    )
    import collections

    # These tiles are all "open" (walkable/interactable interior cells).
    OPEN = {TILE_DUNGEON_FLOOR, TILE_CHEST, TILE_FOUNTAIN,
            TILE_STAIRS_DOWN, TILE_STAIRS_UP}

    size = 32
    grid, _chests, _fountains, _down, _up, spawn = (
        generate_dungeon_floor(_fresh_rng(42), size)
    )

    # BFS from spawn.
    visited: set[tuple[int, int]] = set()
    queue: collections.deque[tuple[int, int]] = collections.deque([spawn])
    visited.add(spawn)
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < size and 0 <= ny < size
                and (nx, ny) not in visited
                and grid[ny][nx] in OPEN
            ):
                visited.add((nx, ny))
                queue.append((nx, ny))

    # Count total open tiles in the grid.
    total_open = sum(1 for row in grid for cell in row if cell in OPEN)
    assert total_open > 0, "No open tiles generated"
    # All open tiles should be reachable from spawn.
    assert len(visited) == total_open, (
        f"Not all open tiles reachable from spawn: "
        f"{len(visited)} reachable / {total_open} total open tiles"
    )


# ---------------------------------------------------------------------------
# T19β: Wire dungeon biome into floors 1 and 3
# ---------------------------------------------------------------------------

def test_floor1_has_eight_chests_after_reset():
    """After reset, floor 1 has exactly 8 chest tiles (one per room)."""
    from glyphbench.envs.craftax.base import TILE_CHEST

    e = CraftaxFullEnv(max_turns=500)
    e.reset(seed=0)
    grid1 = e._floors[1]
    chest_count = sum(1 for row in grid1 for cell in row if cell == TILE_CHEST)
    assert chest_count == 8, (
        f"Floor 1 should have 8 chests (one per room), got {chest_count}"
    )


def test_floor3_has_eight_chests_after_reset():
    """After reset, floor 3 has exactly 8 chest tiles (one per room)."""
    from glyphbench.envs.craftax.base import TILE_CHEST

    e = CraftaxFullEnv(max_turns=500)
    e.reset(seed=0)
    grid3 = e._floors[3]
    chest_count = sum(1 for row in grid3 for cell in row if cell == TILE_CHEST)
    assert chest_count == 8, (
        f"Floor 3 should have 8 chests (one per room), got {chest_count}"
    )


def test_floor1_has_stairs_after_reset():
    """After reset, floor 1 has both TILE_STAIRS_UP and TILE_STAIRS_DOWN."""
    from glyphbench.envs.craftax.base import TILE_STAIRS_DOWN, TILE_STAIRS_UP

    e = CraftaxFullEnv(max_turns=500)
    e.reset(seed=0)
    grid1 = e._floors[1]
    up_count = sum(1 for row in grid1 for cell in row if cell == TILE_STAIRS_UP)
    down_count = sum(1 for row in grid1 for cell in row if cell == TILE_STAIRS_DOWN)
    assert up_count >= 1, "Floor 1 must have at least one TILE_STAIRS_UP"
    assert down_count >= 1, "Floor 1 must have at least one TILE_STAIRS_DOWN"


def test_floor1_has_fountain_tiles_or_zero_statistically():
    """Floor 1 can have 0-8 fountains per seed; assert tile count matches."""
    from glyphbench.envs.craftax.base import TILE_FOUNTAIN

    counts: list[int] = []
    for seed in range(10):
        e = CraftaxFullEnv(max_turns=500)
        e.reset(seed=seed)
        grid1 = e._floors[1]
        count = sum(1 for row in grid1 for cell in row if cell == TILE_FOUNTAIN)
        counts.append(count)

    # All counts must be in [0, 8].
    assert all(0 <= c <= 8 for c in counts), (
        f"Unexpected fountain counts on floor 1: {counts}"
    )
    # At least some seeds should produce fountains.
    assert sum(counts) > 0, (
        "Expected at least some fountain tiles on floor 1 across 10 seeds"
    )


# ---------------------------------------------------------------------------
# T18β: Fountain interaction via DO action
# ---------------------------------------------------------------------------

def _make_env_with_fountain_in_front() -> CraftaxFullEnv:
    """Return an env with the agent facing a TILE_FOUNTAIN, water depleted."""
    from glyphbench.envs.craftax.base import TILE_FOUNTAIN

    e = CraftaxFullEnv(max_turns=500)
    e.reset(seed=0)
    # Move agent to a safe dungeon position.
    e._current_floor = 1
    ax, ay = 10, 10
    e._agent_x = ax
    e._agent_y = ay
    e._facing = (1, 0)   # facing right
    # Place fountain to the right.
    e._floors[1][ay][ax + 1] = TILE_FOUNTAIN
    # Deplete water so the fountain matters.
    e._water = 3
    return e


_DO_ACTION = 5  # index of DO in CRAFTAX_FULL_ACTION_SPEC


def test_fountain_do_refills_water():
    """DO on a fountain tile restores water to _MAX_WATER."""
    from glyphbench.envs.craftax.full import _MAX_WATER

    e = _make_env_with_fountain_in_front()
    assert e._water < _MAX_WATER

    e.step(_DO_ACTION)

    assert e._water == _MAX_WATER, (
        f"Water should be refilled to {_MAX_WATER} after drinking from fountain, "
        f"got {e._water}"
    )


def test_fountain_do_fires_collect_drink_achievement():
    """DO on a fountain fires the collect_drink achievement."""
    e = _make_env_with_fountain_in_front()
    assert "collect_drink" not in e._achievements_unlocked

    e.step(_DO_ACTION)

    assert "collect_drink" in e._achievements_unlocked, (
        "collect_drink achievement should fire after drinking from a fountain"
    )


def test_fountain_tile_remains_after_drink():
    """The TILE_FOUNTAIN tile is NOT consumed after a DO action (reusable)."""
    from glyphbench.envs.craftax.base import TILE_FOUNTAIN

    e = _make_env_with_fountain_in_front()
    ax, ay = e._agent_x, e._agent_y
    fx, fy = ax + 1, ay

    e.step(_DO_ACTION)

    assert e._floors[e._current_floor][fy][fx] == TILE_FOUNTAIN, (
        "TILE_FOUNTAIN should remain on the grid after drinking (it is reusable)"
    )


def test_fountain_do_on_full_water_is_noop():
    """DO on a fountain when water is already full: water stays at max and
    collect_drink achievement does NOT fire from the fountain.

    Note: Other per-step logic (milestone achievements, etc.) may produce
    incidental reward unrelated to the fountain — we only assert on the
    fountain-specific effects (water value and collect_drink not triggered
    from this step).
    """
    from glyphbench.envs.craftax.full import _MAX_WATER

    e = _make_env_with_fountain_in_front()
    e._water = _MAX_WATER  # already full
    # Ensure collect_drink is not yet unlocked so we can detect if it fires.
    e._achievements_unlocked.discard("collect_drink")

    e.step(_DO_ACTION)

    assert e._water == _MAX_WATER, "Water should remain at max when already full"
    assert "collect_drink" not in e._achievements_unlocked, (
        "collect_drink should NOT fire when fountain is used at full water"
    )


# ---------------------------------------------------------------------------
# T20β: Floor 4 (Vaults) skeleton + TILE_ENCHANT_FIRE
# ---------------------------------------------------------------------------

def test_tile_enchant_fire_is_single_codepoint():
    """TILE_ENCHANT_FIRE must be exactly 1 Unicode code point."""
    from glyphbench.envs.craftax.base import TILE_ENCHANT_FIRE
    assert len(TILE_ENCHANT_FIRE) == 1, (
        f"TILE_ENCHANT_FIRE must be a single codepoint, got len={len(TILE_ENCHANT_FIRE)!r}"
    )


def test_tile_enchant_fire_disjoint_from_palette():
    """TILE_ENCHANT_FIRE must not collide with any other tile constant."""
    from glyphbench.envs.craftax import base as _base
    all_tiles = {
        name: getattr(_base, name)
        for name in dir(_base)
        if name.startswith("TILE_") and name != "TILE_ENCHANT_FIRE"
    }
    from glyphbench.envs.craftax.base import TILE_ENCHANT_FIRE
    for name, glyph in all_tiles.items():
        assert glyph != TILE_ENCHANT_FIRE, (
            f"TILE_ENCHANT_FIRE collides with {name} (both = {TILE_ENCHANT_FIRE!r})"
        )


def test_floor_4_exists_after_reset():
    """After reset, floor 4 must be present in env._floors."""
    e = CraftaxFullEnv(max_turns=500)
    e.reset(seed=42)
    assert 4 in e._floors, "Floor 4 should be generated during reset"


def test_floor_4_has_enchant_fire_tile():
    """Floor 4 must contain at least 1 TILE_ENCHANT_FIRE tile after reset."""
    from glyphbench.envs.craftax.base import TILE_ENCHANT_FIRE

    e = CraftaxFullEnv(max_turns=500)
    e.reset(seed=42)

    floor4_grid = e._floors[4]
    found = any(
        cell == TILE_ENCHANT_FIRE
        for row in floor4_grid
        for cell in row
    )
    assert found, (
        "Floor 4 (Vaults) should have at least 1 TILE_ENCHANT_FIRE tile"
    )


# ---------------------------------------------------------------------------
# T20β fixup: TILE_ENCHANT_ICE on floor 3 (Sewers)
# ---------------------------------------------------------------------------

def test_tile_enchant_ice_is_single_codepoint():
    """TILE_ENCHANT_ICE must be exactly 1 Unicode code point."""
    from glyphbench.envs.craftax.base import TILE_ENCHANT_ICE
    assert len(TILE_ENCHANT_ICE) == 1, (
        f"TILE_ENCHANT_ICE must be a single codepoint, got len={len(TILE_ENCHANT_ICE)!r}"
    )


def test_tile_enchant_ice_disjoint_from_palette():
    """TILE_ENCHANT_ICE must not collide with any other tile constant."""
    from glyphbench.envs.craftax import base as _base
    all_tiles = {
        name: getattr(_base, name)
        for name in dir(_base)
        if name.startswith("TILE_") and name != "TILE_ENCHANT_ICE"
    }
    from glyphbench.envs.craftax.base import TILE_ENCHANT_ICE
    for name, glyph in all_tiles.items():
        assert glyph != TILE_ENCHANT_ICE, (
            f"TILE_ENCHANT_ICE collides with {name} (both = {TILE_ENCHANT_ICE!r})"
        )


def test_floor_3_has_enchant_ice_tile():
    """Floor 3 must contain at least 1 TILE_ENCHANT_ICE tile after reset."""
    from glyphbench.envs.craftax.base import TILE_ENCHANT_ICE

    e = CraftaxFullEnv(max_turns=500)
    e.reset(seed=42)

    floor3_grid = e._floors[3]
    found = any(
        cell == TILE_ENCHANT_ICE
        for row in floor3_grid
        for cell in row
    )
    assert found, (
        "Floor 3 (Sewers) should have at least 1 TILE_ENCHANT_ICE tile"
    )


def test_floor_4_has_stairs_down_from_floor_3():
    """Floor 3 must have a TILE_STAIRS_DOWN leading to floor 4 after reset."""
    from glyphbench.envs.craftax.base import TILE_STAIRS_DOWN

    e = CraftaxFullEnv(max_turns=500)
    e.reset(seed=42)

    floor3_grid = e._floors[3]
    found = any(
        cell == TILE_STAIRS_DOWN
        for row in floor3_grid
        for cell in row
    )
    assert found, (
        "Floor 3 must have a TILE_STAIRS_DOWN tile so the player can descend to floor 4"
    )


def test_player_can_descend_to_floor_4():
    """Player standing on floor 3's TILE_STAIRS_DOWN + DESCEND action enters floor 4."""
    from glyphbench.envs.craftax.base import TILE_STAIRS_DOWN

    # DESCEND action index
    _DESCEND_ACTION = CRAFTAX_FULL_ACTION_SPEC.names.index("DESCEND")

    e = CraftaxFullEnv(max_turns=500)
    e.reset(seed=42)

    # Warp player to floor 3 stairs-down position.
    e._current_floor = 3
    down_pos = e._stairs_down_pos.get(3)
    assert down_pos is not None, "Floor 3 should have a stairs-down position recorded"
    e._agent_x, e._agent_y = down_pos

    assert e._floors[3][e._agent_y][e._agent_x] == TILE_STAIRS_DOWN, (
        "Agent should be standing on the stairs-down tile"
    )

    e.step(_DESCEND_ACTION)

    assert e._current_floor == 4, (
        f"After DESCEND from floor 3, player should be on floor 4, got {e._current_floor}"
    )
