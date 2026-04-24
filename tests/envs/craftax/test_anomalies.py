"""Regression tests for Craftax anomalies flagged in review-2026-04-21.

Documents invariants for the 6 anomalous subtasks (fight-skeletons,
fight-cow, fight-archers, survive-night, speedrun, survivehorde) and
verifies the turn-budget bumps stuck (craftchain, dungeonclear,
gatherresources, choptrees, minestone).

These are *invariant* tests, not balance fixes.  The anomalies were
investigated and found to be design balance issues (random policy dies
fast vs. many mobs, or gets achievement-auto-unlocks for having
endgame gear), not code bugs.  See
``runs/review-2026-04-21/craftax-followup.md`` for details.
"""
from __future__ import annotations

from glyphbench.core import make_env
import glyphbench.envs.craftax  # register envs


import pytest

import glyphbench  # noqa: F401 -- registers envs


# ---------------------------------------------------------------------------
# Finding 1: turn budgets bumped
# ---------------------------------------------------------------------------

BUDGET_EXPECTATIONS = {
    "glyphbench/craftax-craftchain-v0": 120,
    "glyphbench/craftax-dungeonclear-v0": 120,
    "glyphbench/craftax-gatherresources-v0": 110,
    "glyphbench/craftax-choptrees-v0": 70,
    "glyphbench/craftax-minestone-v0": 70,
}


@pytest.mark.parametrize("env_id,expected", list(BUDGET_EXPECTATIONS.items()))
def test_budget_bumps_stick(env_id: str, expected: int) -> None:
    env = make_env(env_id)
    assert env.max_turns == expected, (
        f"{env_id} max_turns={env.max_turns} != expected {expected}"
    )


# ---------------------------------------------------------------------------
# Finding 2: anomaly invariants (mobs *do* spawn; rewards *do* fire)
# ---------------------------------------------------------------------------

def _reset(env_id: str, seed: int = 0):
    env = make_env(env_id)
    env.reset(seed)
    return env


def test_fight_skeletons_mobs_spawn() -> None:
    """3 skeletons always spawn at reset -- avg_len 5.6 is player dying,
    not empty arena with instant termination."""
    for seed in [0, 1, 2, 3, 4]:
        env = _reset("glyphbench/craftax-fight-skeletons-v0", seed)
        skels = [m for m in env._mobs if m["type"] == "skeleton"]
        assert len(skels) == 3, f"seed={seed}: expected 3 skeletons, got {len(skels)}"


def test_fight_archers_mobs_spawn() -> None:
    """3 skeleton archers always spawn at reset."""
    for seed in [0, 1, 2, 3, 4]:
        env = _reset("glyphbench/craftax-fight-archers-v0", seed)
        archers = [m for m in env._mobs if m["type"] == "skeleton_archer"]
        assert len(archers) == 3, f"seed={seed}: expected 3 archers, got {len(archers)}"


def test_survivehorde_mobs_spawn() -> None:
    """5 hostile mobs (3 zombies + 2 skeletons) always spawn."""
    for seed in [0, 1, 2, 3, 4]:
        env = _reset("glyphbench/craftax-survivehorde-v0", seed)
        z = sum(1 for m in env._mobs if m["type"] == "zombie")
        k = sum(1 for m in env._mobs if m["type"] == "skeleton")
        assert z == 3 and k == 2, f"seed={seed}: zombies={z}, skeletons={k}"


def test_fight_cow_reward_fires_on_kill() -> None:
    """Scripted: pin the cow adjacent and DO it down to 0 HP.
    Verifies +5 reward + episode termination fires on cow death."""
    env = _reset("glyphbench/craftax-fight-cow-v0", seed=0)
    cow = next(m for m in env._mobs if m["type"] == "cow")
    env._agent_x = cow["x"] - 1
    env._agent_y = cow["y"]
    env._facing = (1, 0)
    do_idx = env.action_spec.names.index("DO")
    total_r = 0.0
    terminated = False
    # Wood sword deals 2 damage; cow has 3 HP -> 2 hits.  Cow does a random
    # walk, so re-pin each step.
    for _ in range(20):
        # Re-pin: cow to (agent_x+1, agent_y)
        cow_mob = next((m for m in env._mobs if m["type"] == "cow"), None)
        if cow_mob is None:
            break
        cow_mob["x"] = env._agent_x + 1
        cow_mob["y"] = env._agent_y
        _, r, terminated, _, _ = env.step(do_idx)
        total_r += r
        if terminated:
            break
    assert terminated, "episode should terminate when cow dies"
    assert total_r >= 5.0, f"expected +5 kill reward, got {total_r}"


def test_survive_night_transitions_to_night_quickly() -> None:
    """Env sets day_counter to DAY_LENGTH - 2, so night starts within 2 steps."""
    env = _reset("glyphbench/craftax-survive-night-v0", seed=0)
    assert env._day_night == "day", "starts in day"
    # Step until night
    noop_idx = 0  # NOOP
    for step in range(5):
        env.step(noop_idx)
        if env._day_night == "night":
            assert step <= 2, f"night took {step+1} steps, expected <= 2"
            return
    pytest.fail("night phase never triggered")


def test_speedrun_starts_on_dungeon_entrance() -> None:
    """Speedrun places the agent at floor-0 stairs-down tile."""
    env = _reset("glyphbench/craftax-speedrun-v0", seed=0)
    entrance = env._stairs_down_pos.get(0)
    assert entrance is not None, "floor 0 must have stairs down"
    assert (env._agent_x, env._agent_y) == entrance, (
        f"agent at ({env._agent_x}, {env._agent_y}) != entrance {entrance}"
    )


def test_speedrun_descends_grants_floor_reward() -> None:
    """When _current_floor increments, +3 reward fires."""
    env = _reset("glyphbench/craftax-speedrun-v0", seed=0)
    # Simulate floor change by patching state -- verifies the _step reward path.
    env._current_floor = 1
    # Any valid action; NOOP keeps state stable, reward path triggers.
    noop_idx = 0
    _, r, _, _, _ = env.step(noop_idx)
    # +3 per floor gained (from 0 to 1).  Additional achievement rewards may
    # also fire; we only assert at least the floor bonus.
    assert r >= 3.0, f"expected at least +3 for new floor, got {r}"
