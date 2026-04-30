"""Phase-β tests: T01β — upstream achievement bitmap state field."""

from __future__ import annotations

import pytest

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

_REST_ACTION = 36   # index of REST in CRAFTAX_FULL_ACTION_SPEC
_NOOP_ACTION = 0    # index of NOOP


def test_rest_action_in_spec():
    """REST is in CRAFTAX_FULL_ACTION_SPEC at index 36 (spec is now 37 actions)."""
    from glyphbench.envs.craftax.base import CRAFTAX_FULL_ACTION_SPEC
    assert len(CRAFTAX_FULL_ACTION_SPEC.names) == 37
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
