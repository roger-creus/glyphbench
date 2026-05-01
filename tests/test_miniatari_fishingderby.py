"""Smoke + win/loss path tests for miniatari-fishingderby-v0."""
from __future__ import annotations

import random

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-fishingderby-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-fishingderby-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-fishingderby-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_fishingderby_win_path() -> None:
    """Random rollout under seed=28 with action-rng (28*7+1) catches 3 fish first.

    Fish drift, opp AI, and reel cycles all use rng, so a deterministic
    forced plan is brittle. We pin a known winning rollout instead.
    Asserts terminated, won=True, 0 < cumulative <= 1.0.
    """
    env = make_env("glyphbench/miniatari-fishingderby-v0")
    env.reset(seed=28)
    action_rng = random.Random(28 * 7 + 1)
    total = 0.0
    won = False
    terminated = False
    for _ in range(env.max_turns):
        a = action_rng.randrange(env.action_spec.n)
        _obs, r, terminated, _trunc, info = env.step(a)
        total += float(r)
        if terminated:
            won = bool(info.get("won", False))
            break
    assert terminated, "win path failed to terminate"
    assert won, f"win path did not win (total={total})"
    assert 0.0 < total <= 1.0 + 1e-6, f"win path total {total} out of (0, 1]"


def test_fishingderby_loss_path() -> None:
    """NOOP-only on seed=0 lets the opponent catch 3 fish first.

    Asserts terminated, won=False, cumulative >= -1.0.
    """
    env = make_env("glyphbench/miniatari-fishingderby-v0")
    env.reset(seed=0)
    NOOP = env.action_spec.names.index("NOOP")
    total = 0.0
    won = True
    terminated = False
    for _ in range(env.max_turns):
        _obs, r, terminated, _trunc, info = env.step(NOOP)
        total += float(r)
        if terminated:
            won = bool(info.get("won", False))
            break
    assert terminated, "loss path did not terminate"
    assert not won, "loss path unexpectedly won"
    assert total >= -1.0 - 1e-6, f"loss total {total} below -1.0"
    assert total <= 1.0 + 1e-6, f"loss total {total} above 1.0"
