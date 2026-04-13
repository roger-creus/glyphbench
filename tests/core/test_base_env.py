import pytest

from rl_world_ascii.core.action import ActionSpec
from rl_world_ascii.core.base_env import BaseAsciiEnv
from rl_world_ascii.core.observation import GridObservation


class _TinyEnv(BaseAsciiEnv):
    """Minimal concrete subclass used only to exercise BaseAsciiEnv's contract."""

    action_spec = ActionSpec(names=("NOOP", "TICK"), descriptions=("", ""))

    def env_id(self) -> str:
        return "rl_world_ascii/__tiny-v0"

    def system_prompt(self) -> str:
        return "You are testing BaseAsciiEnv. Do anything."

    def _reset(self, seed: int) -> GridObservation:
        self._counter = 0
        return self._build_obs()

    def _step(self, action: int):
        if action == 1:  # TICK
            self._counter += 1
        reward = 1.0 if self._counter == 3 else 0.0
        terminated = self._counter >= 3
        return self._build_obs(), reward, terminated, False, {"counter": self._counter}

    def _render_current_observation(self) -> GridObservation:
        return self._build_obs()

    def _build_obs(self) -> GridObservation:
        return GridObservation(
            grid=f"counter={self._counter}",
            legend="(none)",
            hud=f"Step: {self._turn}",
            message="",
        )


def test_reset_requires_explicit_seed():
    env = _TinyEnv()
    with pytest.raises(ValueError, match="seed"):
        env.reset()  # type: ignore[call-arg]


def test_reset_returns_rendered_string_and_info_with_turn_zero():
    env = _TinyEnv()
    obs, info = env.reset(seed=42)
    assert isinstance(obs, str)
    assert "counter=0" in obs
    assert info["turn"] == 0
    assert info["seed"] == 42
    assert info["env_id"] == "rl_world_ascii/__tiny-v0"


def test_reset_determinism_same_seed_same_initial_obs():
    env1 = _TinyEnv()
    env2 = _TinyEnv()
    o1, _ = env1.reset(seed=7)
    o2, _ = env2.reset(seed=7)
    assert o1 == o2


def test_step_returns_five_tuple_and_increments_turn():
    env = _TinyEnv()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(1)
    assert isinstance(obs, str)
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info["turn"] == 1
    assert info["counter"] == 1


def test_step_terminates_at_counter_three_with_reward_one():
    env = _TinyEnv()
    env.reset(seed=0)
    env.step(1)
    env.step(1)
    obs, reward, terminated, truncated, info = env.step(1)
    assert terminated
    assert reward == 1.0


def test_step_rejects_non_integer_action():
    env = _TinyEnv()
    env.reset(seed=0)
    with pytest.raises(TypeError):
        env.step("TICK")  # type: ignore[arg-type]


def test_step_rejects_bool_action():
    env = _TinyEnv()
    env.reset(seed=0)
    with pytest.raises(TypeError):
        env.step(True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        env.step(False)  # type: ignore[arg-type]


def test_step_rejects_out_of_range_action():
    env = _TinyEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(99)


def test_max_turns_truncation():
    env = _TinyEnv(max_turns=2)
    env.reset(seed=0)
    env.step(0)  # NOOP
    obs, reward, terminated, truncated, info = env.step(0)  # NOOP, turn 2 -> truncate
    assert truncated
    assert info.get("truncation_reason") == "max_turns"


def test_rng_is_numpy_generator_and_seeded():
    env = _TinyEnv()
    env.reset(seed=123)
    r1 = env.rng.integers(0, 10_000)
    env.reset(seed=123)
    r2 = env.rng.integers(0, 10_000)
    assert r1 == r2


def test_rng_before_reset_raises():
    env = _TinyEnv()
    with pytest.raises(RuntimeError):
        _ = env.rng
