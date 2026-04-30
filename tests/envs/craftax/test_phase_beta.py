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
