"""Smoke + win/loss path tests for miniatari-kungfumaster-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-kungfumaster-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-kungfumaster-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-kungfumaster-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_kungfumaster_win_path() -> None:
    """KICK spam wins on seed=0: each KICK strikes within 2 cells on both sides.

    With 8 enemies spawned alternately from the edges, standing at the
    centre and kicking every tick takes them out as they enter range.
    Asserts terminated, won=True, 0 < cumulative <= 1.0.
    """
    env = make_env("glyphbench/miniatari-kungfumaster-v0")
    env.reset(seed=0)
    KICK = env.action_spec.names.index("KICK")
    total = 0.0
    won = False
    terminated = False
    for _ in range(env.max_turns):
        _obs, r, terminated, _trunc, info = env.step(KICK)
        total += float(r)
        if terminated:
            won = bool(info.get("won", False))
            break
    assert terminated, "win path failed to terminate"
    assert won, f"win path did not win (total={total})"
    assert 0.0 < total <= 1.0 + 1e-6, f"win path total {total} out of (0, 1]"


def test_kungfumaster_loss_path() -> None:
    """NOOP-only lets the first enemy reach the player on seed=0.

    Asserts terminated, won=False, cumulative >= -1.0.
    """
    env = make_env("glyphbench/miniatari-kungfumaster-v0")
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
