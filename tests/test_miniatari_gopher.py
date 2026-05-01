"""Smoke + win/loss path tests for miniatari-gopher-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-gopher-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-gopher-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-gopher-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_gopher_win_path() -> None:
    """Active-defense path: walk to gopher's column and FILL forever (seed=0).

    On seed=0 the gopher starts targeting column 11. The player walks 4
    RIGHT to reach col 11, then FILLs each tick to keep dirt above 0.
    Asserts terminated, won=True, 0 < cumulative <= 1.0.
    """
    env = make_env("glyphbench/miniatari-gopher-v0")
    env.reset(seed=0)
    names = env.action_spec.names
    RIGHT = names.index("RIGHT")
    FILL = names.index("FILL")
    plan = [RIGHT] * 4 + [FILL] * 60
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


def test_gopher_loss_path() -> None:
    """NOOP-only loses all 3 carrots and emits -1 terminal (seed=0).

    Asserts terminated, won=False, cumulative >= -1.0 and == -1.0 on full loss.
    """
    env = make_env("glyphbench/miniatari-gopher-v0")
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
