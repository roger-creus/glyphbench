"""Smoke + win/loss path tests for miniatari-breakout-v0."""
from __future__ import annotations

import numpy as np

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-breakout-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-breakout-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-breakout-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_breakout_random_play_breaks_bricks() -> None:
    """Brute-force regression: under random play, at least 50% of seeds
    should break at least one brick within 300 steps. Original 6-brick
    layout at cols 5..10 with parity-1 ball trajectory hit zero bricks
    across 100 seeds."""
    hits = 0
    for seed in range(100):
        env = make_env("glyphbench/miniatari-breakout-v0")
        env.reset(seed=seed)
        rng = np.random.default_rng(seed + 1000)
        bricks_init = len(env._bricks)
        for _ in range(300):
            a = int(rng.integers(0, env.action_spec.n))
            _, _, terminated, truncated, _ = env.step(a)
            if terminated or truncated:
                break
        if len(env._bricks) < bricks_init:
            hits += 1
    assert hits >= 50, f"Only {hits}/100 seeds broke any brick"


def test_breakout_win_path() -> None:
    """Paddle-tracking on seed=0 breaks 11 of 14 bricks → win. The
    residual cells (col 3 or cols 7+8 depending on serve direction) are
    parity-unreachable, so the win threshold is set to 11."""
    env = make_env("glyphbench/miniatari-breakout-v0")
    env.reset(seed=0)
    left = env.action_spec.names.index("LEFT")
    right = env.action_spec.names.index("RIGHT")
    noop = env.action_spec.names.index("NOOP")
    cumulative = 0.0
    won = False
    terminated = False
    for _ in range(env.max_turns):
        paddle_center = env._paddle_x + env._PADDLE_W // 2
        if env._ball_x < paddle_center:
            action = left
        elif env._ball_x > paddle_center:
            action = right
        else:
            action = noop
        _, reward, terminated, truncated, info = env.step(action)
        cumulative += reward
        if terminated or truncated:
            won = bool(info.get("won"))
            break
    assert terminated, "expected paddle-tracking to terminate via win"
    assert won, f"expected won=True, got won={won} cumulative={cumulative}"
    assert 0.0 < cumulative <= 1.0 + 1e-6


def test_breakout_loss_path() -> None:
    """NOOP only: ball serves at (8, 7) heading up, bounces around, and
    eventually falls past the paddle (which never moves to intercept it
    under stuck-centre serve)."""
    env = make_env("glyphbench/miniatari-breakout-v0")
    env.reset(seed=0)
    noop = env.action_spec.names.index("NOOP")
    cumulative = 0.0
    won = True
    terminated = False
    for _ in range(env.max_turns):
        _, reward, terminated, truncated, info = env.step(noop)
        cumulative += reward
        if terminated or truncated:
            won = bool(info.get("won"))
            break
    assert terminated or truncated
    if terminated:
        assert not won
    assert -1.0 - 1e-6 <= cumulative <= 1.0 + 1e-6
