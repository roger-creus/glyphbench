"""Behavioural tests for the curated Craftax sub-tasks.

These tests don't try to brute-force the full crafting chain — they
verify the tighter contracts: NOOP-spam should NOT win the survival /
bootstrap envs, success-on-condition rewards fire when their guard
condition holds, and the recently-fixed death-penalty branch in the
SubtaskMixin combat envs is now reachable.
"""

from __future__ import annotations

import pytest

from glyphbench.core.registry import make_env
from glyphbench.envs.craftax.classic import CraftaxClassicEnv


def _run(env: CraftaxClassicEnv, action: int, n: int) -> tuple[float, bool, bool]:
    """Step ``action`` ``n`` times. Return cumulative reward and the
    final (terminated, truncated) flags."""
    total = 0.0
    terminated = False
    truncated = False
    for _ in range(n):
        if terminated or truncated:
            break
        _, r, terminated, truncated, _ = env.step(action)
        total += r
    return total, terminated, truncated


def _action_idx(env: CraftaxClassicEnv, name: str) -> int:
    return env.action_spec.index_of(name)


# ---------------------------------------------------------------------------
# survive-hunger / survive-wild — NOOP-spam should not win
# ---------------------------------------------------------------------------


def test_survive_hunger_noop_dies_before_max_turns():
    env = make_env("glyphbench/craftax-survive-hunger-v0")
    env.reset(seed=0)
    noop = _action_idx(env, "NOOP")
    total, terminated, truncated = _run(env, noop, env.max_turns)
    assert terminated, "agent should die from starvation"
    assert not truncated, "should not have hit the truncation cap"
    # Food=1, drain at step 50 → 0, damage at step 51, dies ~step 60.
    assert env._turn < env.max_turns


def test_survive_hunger_no_food_no_bonus():
    env = make_env("glyphbench/craftax-survive-hunger-v0")
    env.reset(seed=0)
    noop = _action_idx(env, "NOOP")
    total, _, _ = _run(env, noop, env.max_turns)
    # Even if the agent somehow survived, the +10 truncation bonus is
    # gated on _food_eaten >= 1.
    assert total < 10.0


def test_survive_wild_nightfall_happens_in_episode():
    env = make_env("glyphbench/craftax-survive-wild-v0")
    env.reset(seed=0)
    # Day counter starts ~30 steps before nightfall (DAY_LENGTH=200).
    assert env._day_counter >= 150
    noop = _action_idx(env, "NOOP")
    for _ in range(40):
        env.step(noop)
        if env._day_night == "night":
            break
    assert env._day_night == "night", "night should fall within the first ~30 steps"


# ---------------------------------------------------------------------------
# Iron / diamond bootstrap — empty-inventory NOOP cannot terminate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_id", [
    "glyphbench/craftax-iron-bootstrap-v0",
    "glyphbench/craftax-diamond-bootstrap-v0",
])
def test_bootstrap_noop_does_not_terminate(env_id: str):
    env = make_env(env_id)
    env.reset(seed=0)
    noop = _action_idx(env, "NOOP")
    total, terminated, truncated = _run(env, noop, env.max_turns)
    assert not terminated or env._hp <= 0, (
        "bootstrap should not be solvable by NOOP; if terminated, "
        "must be from death rather than a success bonus"
    )
    assert env._inventory.get("iron", 0) == 0
    assert env._inventory.get("diamond", 0) == 0


def test_iron_bootstrap_starting_inventory_empty():
    env = make_env("glyphbench/craftax-iron-bootstrap-v0")
    env.reset(seed=0)
    assert env._inventory == {}


def test_diamond_bootstrap_seed_layout_has_diamond_tile():
    env = make_env("glyphbench/craftax-diamond-bootstrap-v0")
    env.reset(seed=0)
    diamonds = sum(row.count("D") for row in env._world)
    assert diamonds >= 1


# ---------------------------------------------------------------------------
# Wave defense — first wave spawns by step 1, NOOP usually loses
# ---------------------------------------------------------------------------


def test_wave_defense_first_wave_spawns_by_step_1():
    env = make_env("glyphbench/craftax-wave-defense-v0")
    env.reset(seed=0)
    assert len(env._mobs) == 0
    noop = _action_idx(env, "NOOP")
    env.step(noop)
    hostiles = [m for m in env._mobs if m["type"] != "cow"]
    assert len(hostiles) >= 1, "wave 1 should have spawned by step 1"


def test_wave_defense_noop_takes_damage():
    env = make_env("glyphbench/craftax-wave-defense-v0")
    env.reset(seed=0)
    noop = _action_idx(env, "NOOP")
    start_hp = env._hp
    for _ in range(20):
        env.step(noop)
    assert env._hp < start_hp, (
        "zombies should reach the agent and damage it on NOOP-spam"
    )


# ---------------------------------------------------------------------------
# Death-penalty branch — fight-zombie now actually applies -10 on death
# ---------------------------------------------------------------------------


def test_fight_zombie_death_penalty_is_reachable():
    """Previously the -10 death branch in CraftaxFightZombieEnv was
    unreachable because _SubtaskMixin._step only ran _subtask_check
    when the parent had not already terminated the episode. After the
    fix, killing the agent should produce a -10 component."""
    env = make_env("glyphbench/craftax-fightzombie-v0")
    env.reset(seed=0)
    # Force the agent to 1 HP and make the zombie adjacent so it dies
    # on the next mob AI tick.
    env._hp = 1
    # Move the zombie to (agent_x+1, agent_y) — directly adjacent.
    if env._mobs:
        env._mobs[0]["x"] = env._agent_x + 1
        env._mobs[0]["y"] = env._agent_y
    noop = _action_idx(env, "NOOP")
    _, r, terminated, _, _ = env.step(noop)
    assert terminated
    # Death branch returns -10. Parent reward on death is 0, so total
    # step reward should be -10.
    assert r == pytest.approx(-10.0, abs=0.5)
