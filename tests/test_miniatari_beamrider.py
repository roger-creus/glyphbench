"""Smoke + win/loss path tests for miniatari-beamrider-v0."""
from __future__ import annotations

import glyphbench  # noqa: F401  # registers all envs
from glyphbench.core import make_env


def test_env_constructs():
    env = make_env("glyphbench/miniatari-beamrider-v0")
    obs, info = env.reset(seed=0)
    assert obs is not None
    assert info["env_id"] == "glyphbench/miniatari-beamrider-v0"


def test_env_steps():
    env = make_env("glyphbench/miniatari-beamrider-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_beamrider_win_path() -> None:
    """FIRE every other tick on seed=0 reliably clears 5 enemies."""
    env = make_env("glyphbench/miniatari-beamrider-v0")
    env.reset(seed=0)
    fire = env.action_spec.names.index("FIRE")
    noop = env.action_spec.names.index("NOOP")
    cumulative = 0.0
    won = False
    terminated = False
    for _ in range(120):
        for action in (fire, noop):
            _, reward, terminated, _, info = env.step(action)
            cumulative += reward
            if terminated:
                won = bool(info.get("won"))
                break
        if terminated:
            break
    assert terminated, "expected episode to terminate within budget"
    assert won, "expected win-path to set won=True"
    assert 0.0 < cumulative <= 1.0 + 1e-6, (
        f"cumulative {cumulative} not in (0, 1]"
    )


def test_beamrider_loss_path() -> None:
    """NOOP only -> enemy descends into player beam, terminal -1."""
    env = make_env("glyphbench/miniatari-beamrider-v0")
    env.reset(seed=0)
    noop = env.action_spec.names.index("NOOP")
    cumulative = 0.0
    won = True
    terminated = False
    for _ in range(200):
        _, reward, terminated, _, info = env.step(noop)
        cumulative += reward
        if terminated:
            won = bool(info.get("won"))
            break
    assert terminated, "expected NOOP-only run to terminate (collision)"
    assert not won, "loss path should set won=False"
    assert -1.0 - 1e-6 <= cumulative, (
        f"cumulative {cumulative} below -1"
    )


def test_beamrider_no_initial_collisions() -> None:
    """Pre-seeded enemies must occupy distinct (beam, y) cells (regression)."""
    for s in range(40):
        env = make_env("glyphbench/miniatari-beamrider-v0")
        env.reset(seed=s)
        cells = {(b, y) for b, y in env._enemies}
        assert len(cells) == len(env._enemies), (
            f"seed={s} has duplicate enemy cells: {env._enemies}"
        )
