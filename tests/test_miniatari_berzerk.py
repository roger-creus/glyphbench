"""Smoke + win/loss path tests for miniatari-berzerk-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-berzerk-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-berzerk-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-berzerk-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_berzerk_win_path() -> None:
    """Greedy face+FIRE sequence on seed=25 clears all 5 robots."""
    env = make_env("glyphbench/miniatari-berzerk-v0")
    env.reset(seed=25)
    fire = env.action_spec.names.index("FIRE")
    noop = env.action_spec.names.index("NOOP")
    # Action ints: 0=NOOP, 5=FIRE. Sequence found via greedy planner: at
    # seed=25 the player starts facing UP at (8,5) and the robots line up
    # for repeated NOOP padding then FIRE shots.
    assert noop == 0 and fire == 5
    actions = [5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 0, 5]
    cumulative = 0.0
    won = False
    terminated = False
    for a in actions:
        _, reward, terminated, _, info = env.step(a)
        cumulative += reward
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "expected episode to terminate"
    assert won, f"expected win, got won={won} cumulative={cumulative}"
    assert 0.0 < cumulative <= 1.0 + 1e-6


def test_berzerk_loss_path() -> None:
    """LEFT-only walks straight into the electrified wall: terminal -1."""
    env = make_env("glyphbench/miniatari-berzerk-v0")
    env.reset(seed=0)
    left = env.action_spec.names.index("LEFT")
    cumulative = 0.0
    won = True
    terminated = False
    for _ in range(50):
        _, reward, terminated, _, info = env.step(left)
        cumulative += reward
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "expected loss path to terminate"
    assert not won, "loss path should not be a win"
    assert -1.0 - 1e-6 <= cumulative
