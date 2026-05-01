"""Smoke + win/loss path tests for miniatari-assault-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-assault-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-assault-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-assault-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# Captured action sequence that clears all 5 enemies on seed=0 by
# tracking each enemy's predicted column and FIRE-ing when the turret
# matches. Generated once by a column-prediction bot; pinned for
# determinism (13 steps).
_ASSAULT_WIN_SEQUENCE_SEED0 = [
    "FIRE", "LEFT", "FIRE", "RIGHT", "FIRE", "RIGHT", "RIGHT",
    "FIRE", "RIGHT", "RIGHT", "RIGHT", "FIRE", "FIRE",
]


def test_assault_win_path() -> None:
    """Scripted FIRE pattern clears all 5 enemies on seed=0."""
    env = make_env("glyphbench/miniatari-assault-v0")
    env.reset(seed=0)
    cumulative = 0.0
    won = False
    terminated = False
    info: dict = {}
    for name in _ASSAULT_WIN_SEQUENCE_SEED0:
        action = env.action_spec.names.index(name)
        _, reward, terminated, truncated, info = env.step(action)
        cumulative += reward
        if terminated or truncated:
            won = bool(info.get("won"))
            break
    assert terminated, "win sequence must terminate the episode"
    assert won, "win sequence must set won=True"
    assert 0.0 < cumulative <= 1.0 + 1e-6, (
        f"cumulative {cumulative} not in (0, 1]"
    )


def test_assault_loss_path() -> None:
    """NOOP-only run lets the V formation reach the turret row.

    Assault uses Pattern A (no -1 on death; just terminates), so the
    contract for the loss path is cumulative >= -1 and won=False.
    """
    env = make_env("glyphbench/miniatari-assault-v0")
    env.reset(seed=0)
    noop = env.action_spec.names.index("NOOP")
    cumulative = 0.0
    won = True
    terminated = False
    info: dict = {}
    for _ in range(env.max_turns):
        _, reward, terminated, truncated, info = env.step(noop)
        cumulative += reward
        if terminated or truncated:
            won = bool(info.get("won"))
            break
    assert terminated, "NOOP-only must terminate when an enemy reaches turret row"
    assert not won, "loss path must set won=False"
    assert -1.0 - 1e-6 <= cumulative <= 1.0 + 1e-6, (
        f"cumulative {cumulative} outside [-1, 1]"
    )
