"""Smoke + win/loss path tests for miniatari-frostbite-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-frostbite-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-frostbite-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-frostbite-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_frostbite_win_path() -> None:
    """Force the agent through actions that visit all 4 floes (seed=0).

    Path found by BFS over the (player, floe-state, tick) state space.
    Asserts terminated, won=True, 0 < cumulative <= 1.0.
    """
    env = make_env("glyphbench/miniatari-frostbite-v0")
    env.reset(seed=0)
    names = env.action_spec.names
    plan_names = [
        "NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "NOOP", "NOOP",
        "DOWN", "DOWN",
        "NOOP", "NOOP", "NOOP",
        "DOWN", "NOOP", "DOWN",
        "DOWN", "NOOP",
        "DOWN", "NOOP", "DOWN", "DOWN",
    ]
    plan = [names.index(a) for a in plan_names]
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
    assert won, f"win path did not set won=True (total={total})"
    assert 0.0 < total <= 1.0 + 1e-6, f"win path total {total} out of (0, 1]"


def test_frostbite_loss_path() -> None:
    """Force a loss by stepping DOWN into water (seed=0).

    Asserts terminated, won=False, cumulative >= -1.0.
    """
    env = make_env("glyphbench/miniatari-frostbite-v0")
    env.reset(seed=0)
    names = env.action_spec.names
    DOWN = names.index("DOWN")
    total = 0.0
    won = True
    terminated = False
    for _ in range(20):
        _obs, r, terminated, _trunc, info = env.step(DOWN)
        total += float(r)
        if terminated:
            won = bool(info.get("won", False))
            break
    assert terminated, "loss path did not terminate"
    assert not won, "loss path unexpectedly won"
    assert total >= -1.0 - 1e-6, f"loss total {total} below -1.0"
    assert total <= 1.0 + 1e-6, f"loss total {total} above 1.0"
