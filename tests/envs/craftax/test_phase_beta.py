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
