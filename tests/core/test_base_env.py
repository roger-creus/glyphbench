"""Tests for the non-gym BaseGlyphEnv base class."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation


class _Tiny(BaseGlyphEnv):
    action_spec = ActionSpec(names=("A", "B"), descriptions=("a", "b"))
    noop_action_name = "A"

    def _reset(self, seed: int) -> GridObservation:
        return GridObservation(grid="X", legend="", hud="", message="")

    def _step(self, action: int):
        return GridObservation(grid="X", legend="", hud="", message=""), 0.0, False, False, {}

    def _render_current_observation(self) -> GridObservation:
        return GridObservation(grid="X", legend="", hud="", message="")

    def system_prompt(self) -> str:
        return "sys"

    def env_id(self) -> str:
        return "tiny-v0"


def test_not_gym_subclass():
    # BaseGlyphEnv must NOT inherit from gymnasium.Env; the module must not even
    # need gymnasium to be importable.
    import sys
    assert "gymnasium" not in sys.modules or not any(
        "gym" in cls.__module__.lower() for cls in _Tiny.__mro__ if cls is not object
    )


def test_reset_requires_int_seed():
    env = _Tiny()
    with pytest.raises(TypeError):
        env.reset("not-an-int")  # type: ignore[arg-type]


def test_reset_returns_text_and_info():
    env = _Tiny()
    obs, info = env.reset(42)
    assert isinstance(obs, str)
    assert info["turn"] == 0
    assert info["env_id"] == "tiny-v0"
    assert info["seed"] == 42


def test_step_rejects_bool_and_out_of_range():
    env = _Tiny()
    env.reset(0)
    with pytest.raises(TypeError):
        env.step(True)  # bools are ints — must be rejected
    with pytest.raises(ValueError):
        env.step(99)


def test_step_returns_five_tuple():
    env = _Tiny()
    env.reset(0)
    out = env.step(0)
    assert len(out) == 5
    obs, reward, term, trunc, info = out
    assert isinstance(obs, str)
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)


def test_max_turns_truncates():
    env = _Tiny(max_turns=3)
    env.reset(0)
    env.step(0); env.step(0)
    _, _, term, trunc, info = env.step(0)
    assert trunc is True
    assert info.get("truncation_reason") == "max_turns"


def test_rng_access_requires_reset():
    env = _Tiny()
    with pytest.raises(RuntimeError):
        _ = env.rng
    env.reset(0)
    rng = env.rng
    assert isinstance(rng, np.random.Generator)


def test_close_is_noop_by_default():
    env = _Tiny()
    env.reset(0)
    assert env.close() is None


def test_no_action_observation_space_attrs():
    # These were gymnasium-only; they must not exist on BaseGlyphEnv.
    env = _Tiny()
    env.reset(0)
    assert not hasattr(env, "action_space")
    assert not hasattr(env, "observation_space")
    assert not hasattr(env, "metadata") or not isinstance(getattr(env, "metadata", None), dict) or "render_modes" not in env.metadata  # tolerate subclass metadata, but shouldn't inherit render_modes
