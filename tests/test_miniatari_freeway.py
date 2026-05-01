"""Smoke + win/loss path tests for miniatari-freeway-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-freeway-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-freeway-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-freeway-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_freeway_win_path() -> None:
    """UP spam crosses the highway 4 times on seed=0.

    Asserts terminated, won=True, 0 < cumulative <= 1.0.
    """
    env = make_env("glyphbench/miniatari-freeway-v0")
    env.reset(seed=0)
    UP = env.action_spec.names.index("UP")
    total = 0.0
    won = False
    terminated = False
    for _ in range(env.max_turns):
        _obs, r, terminated, _trunc, info = env.step(UP)
        total += float(r)
        if terminated:
            won = bool(info.get("won", False))
            break
    assert terminated, "win path failed to terminate"
    assert won, f"win path did not win (total={total})"
    assert 0.0 < total <= 1.0 + 1e-6, f"win path total {total} out of (0, 1]"


def test_freeway_loss_path() -> None:
    """NOOP-only never crosses; the episode truncates at max_turns with reward 0.

    Freeway has no terminal-loss mechanic (collisions just bump the chicken
    back), so the loss path here is the natural truncation case.
    Asserts truncation (or termination) with won=False, cumulative >= -1.0
    and cumulative <= 1.0.
    """
    env = make_env("glyphbench/miniatari-freeway-v0")
    env.reset(seed=0)
    NOOP = env.action_spec.names.index("NOOP")
    total = 0.0
    won = True
    ended = False
    for _ in range(env.max_turns + 1):
        _obs, r, terminated, truncated, info = env.step(NOOP)
        total += float(r)
        if terminated or truncated:
            won = bool(info.get("won", False))
            ended = True
            break
    assert ended, "loss path did not end (no termination, no truncation)"
    assert not won, "loss path unexpectedly won"
    assert total >= -1.0 - 1e-6, f"loss total {total} below -1.0"
    assert total <= 1.0 + 1e-6, f"loss total {total} above 1.0"
