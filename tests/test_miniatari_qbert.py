"""Smoke + win/loss path tests for miniatari-qbert-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-qbert-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-qbert-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-qbert-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_qbert_apex_unpainted_at_reset() -> None:
    """Regression: apex must be unpainted at reset so the agent can earn
    +1/10 for painting it. Previously the apex was auto-painted with
    _progress=1 but no reward, capping the achievable cumulative at 0.9."""
    env = make_env("glyphbench/miniatari-qbert-v0")
    env.reset(seed=0)
    assert env._progress == 0
    assert env._painted[0][0] is False


def test_qbert_win_path() -> None:
    """Drive a precomputed 12-action sequence (BFS on the painted-bitmask
    state space) that paints all 10 cubes. Asserts terminated, won=True,
    cumulative == 1.0 (10 cubes * +1/10)."""
    env = make_env("glyphbench/miniatari-qbert-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    actions = [
        "DOWN_LEFT", "UP_RIGHT", "DOWN_RIGHT", "DOWN_RIGHT", "DOWN_RIGHT",
        "UP_LEFT", "DOWN_LEFT", "UP_LEFT", "DOWN_LEFT", "UP_LEFT",
        "DOWN_LEFT", "NOOP",
    ]
    total = 0.0
    terminated = False
    won = False
    for name in actions:
        _, r, terminated, _truncated, info = env.step(ai(name))
        total += r
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "win-path script should reach terminal state"
    assert won, "win-path script should set won=True"
    assert 0.0 < total <= 1.0 + 1e-9
    # The full clear should sum to exactly +1.0 (10 cubes * 1/10).
    assert abs(total - 1.0) < 1e-9


def test_qbert_loss_path() -> None:
    """Hopping UP_LEFT from the apex (row=0, col=0) tries to land at
    (-1, -1) which is off the pyramid. Asserts terminated, won=False,
    cumulative >= -1.0 (no death penalty; just terminates per spec)."""
    env = make_env("glyphbench/miniatari-qbert-v0")
    env.reset(seed=0)
    ai = env.action_spec.names.index
    total = 0.0
    _, r, terminated, _truncated, info = env.step(ai("UP_LEFT"))
    total += r
    assert terminated, "hopping off the pyramid should terminate"
    assert not bool(info.get("won")), "won must be False on a fall"
    assert -1.0 - 1e-9 <= total <= 1.0 + 1e-9
