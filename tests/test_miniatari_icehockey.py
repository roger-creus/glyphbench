"""Smoke + win/loss path tests for miniatari-icehockey-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-icehockey-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-icehockey-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-icehockey-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_icehockey_win_path() -> None:
    """Three UP-UP-UP-SHOOT-NOOP cycles score 3 goals (seed=0).

    Asserts terminated, won=True, 0 < cumulative <= 1.0.
    """
    env = make_env("glyphbench/miniatari-icehockey-v0")
    env.reset(seed=0)
    names = env.action_spec.names
    UP = names.index("UP")
    SHOOT = names.index("SHOOT")
    NOOP = names.index("NOOP")
    plan = ([UP, UP, UP, SHOOT, NOOP]) * 5  # 5 cycles, only 3 needed
    total = 0.0
    won = False
    terminated = False
    for a in plan:
        _obs, r, terminated, _trunc, info = env.step(a)
        total += float(r)
        if terminated:
            won = bool(info.get("won", False))
            break
    assert terminated, "win path failed to terminate"
    assert won, f"win path did not win (total={total})"
    assert 0.0 < total <= 1.0 + 1e-6, f"win path total {total} out of (0, 1]"


def test_icehockey_loss_path() -> None:
    """Walk LEFT then NOOP to let opponent score 3 goals (seed=0).

    Each LEFT*5 + NOOP*15 round lets the opponent collect and shoot the puck.
    Five rounds is enough for 3 opp goals on seed=0.
    Asserts terminated, won=False, cumulative >= -1.0.
    """
    env = make_env("glyphbench/miniatari-icehockey-v0")
    env.reset(seed=0)
    names = env.action_spec.names
    LEFT = names.index("LEFT")
    NOOP = names.index("NOOP")
    plan = ([LEFT] * 5 + [NOOP] * 15) * 6
    total = 0.0
    won = True
    terminated = False
    for a in plan:
        _obs, r, terminated, _trunc, info = env.step(a)
        total += float(r)
        if terminated:
            won = bool(info.get("won", False))
            break
    assert terminated, "loss path did not terminate"
    assert not won, "loss path unexpectedly won"
    assert total >= -1.0 - 1e-6, f"loss total {total} below -1.0"
    assert total <= 1.0 + 1e-6, f"loss total {total} above 1.0"
