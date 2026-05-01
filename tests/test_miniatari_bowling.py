"""Smoke + win/loss path tests for miniatari-bowling-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-bowling-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-bowling-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-bowling-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_bowling_pin_layout_supports_cascade() -> None:
    """Tight pin layout means a centre roll cascades to most of the rack
    (regression test for the original 2-col-gap layout)."""
    env = make_env("glyphbench/miniatari-bowling-v0")
    env.reset(seed=0)
    # The new layout has adjacent columns at every row.
    pins = env._pins
    # Roll at col 5 should knock more than 2 pins thanks to the cascade.
    knocked = env._knock_pins_in_column(5)
    assert knocked >= 5, (
        f"Expected centre roll to knock >=5 pins via cascade, got {knocked}; "
        f"layout was {sorted(pins)}"
    )


def test_bowling_win_path() -> None:
    """Two strikes (one per frame) clear all 20 pins for cumulative=1.0."""
    env = make_env("glyphbench/miniatari-bowling-v0")
    env.reset(seed=0)
    left = env.action_spec.names.index("LEFT")
    right = env.action_spec.names.index("RIGHT")
    fire = env.action_spec.names.index("FIRE")
    # Frame 1: bowler at x=6 → move LEFT to x=5, FIRE (cascade strike: 9
    # pins). Move LEFT twice to x=3, FIRE for the final pin.
    actions = [left, fire, left, left, fire,
               # Frame 2 (pinset reset). Bowler now at x=3. Move RIGHT
               # twice to x=5, FIRE (9 pins). LEFT twice, FIRE for the
               # final pin.
               right, right, fire, left, left, fire]
    cumulative = 0.0
    won = False
    terminated = False
    for a in actions:
        _, reward, terminated, _, info = env.step(a)
        cumulative += reward
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "expected forced win path to terminate"
    assert won, f"expected won=True, got won={won} cumulative={cumulative}"
    assert 0.0 < cumulative <= 1.0 + 1e-6


def test_bowling_loss_path() -> None:
    """No -1 path exists in bowling. Random play stays in [-1, 1] and
    NOOP-only never crosses the win threshold (cumulative stays at 0)."""
    env = make_env("glyphbench/miniatari-bowling-v0")
    env.reset(seed=0)
    noop = env.action_spec.names.index("NOOP")
    cumulative = 0.0
    won = True
    for _ in range(env.max_turns):
        _, reward, terminated, truncated, info = env.step(noop)
        cumulative += reward
        if terminated or truncated:
            won = bool(info.get("won"))
            break
    # NOOP only never fires, so progress stays at 0 and cumulative is 0.0.
    assert not won
    assert -1.0 - 1e-6 <= cumulative <= 1.0 + 1e-6
    # Also assert the bound is tight under random rollout
    import numpy as np
    rng = np.random.default_rng(0)
    env.reset(seed=1)
    cum = 0.0
    for _ in range(200):
        a = int(rng.integers(0, env.action_spec.n))
        _, r, t, tr, _ = env.step(a)
        cum += r
        if t or tr:
            break
    assert -1.0 - 1e-6 <= cum <= 1.0 + 1e-6
