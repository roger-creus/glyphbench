"""Suite-level conformance test for all registered MiniGrid environments.

Iterates over every env matching glyphbench/minigrid-* and validates the contract.
"""

from __future__ import annotations

import gymnasium as gym
import pytest

import glyphbench  # noqa: F401


def _minigrid_env_ids() -> list[str]:
    """Collect all registered glyphbench/minigrid-* env IDs."""
    return sorted(
        eid
        for eid in gym.registry
        if isinstance(eid, str) and eid.startswith("glyphbench/minigrid-")
    )


@pytest.fixture(params=_minigrid_env_ids())
def env_id(request: pytest.FixtureRequest) -> str:
    return request.param


class TestMiniGridSuiteConformance:
    def test_observation_contract(self, env_id: str) -> None:
        for seed in [0, 1, 2]:
            env = gym.make(env_id, max_turns=50)
            obs, info = env.reset(seed=seed)
            assert isinstance(obs, str)
            assert len(obs) > 0
            for _ in range(50):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                assert isinstance(obs, str)
                assert len(obs) > 0
                if terminated or truncated:
                    break

    def test_action_space_is_7(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=10)
        from glyphbench.core.base_env import BaseAsciiEnv
        unwrapped: BaseAsciiEnv = env.unwrapped  # type: ignore[assignment]
        assert unwrapped.action_spec.n == 7

    def test_seed_reproducibility(self, env_id: str) -> None:
        for seed in [0, 42, 99]:
            e1 = gym.make(env_id, max_turns=50)
            e2 = gym.make(env_id, max_turns=50)
            o1, _ = e1.reset(seed=seed)
            o2, _ = e2.reset(seed=seed)
            assert o1 == o2
            for _ in range(10):
                a = 2  # MOVE_FORWARD
                o1, r1, t1, tr1, _ = e1.step(a)
                o2, r2, t2, tr2, _ = e2.step(a)
                assert o1 == o2
                assert r1 == r2
                if t1 or tr1:
                    break

    def test_system_prompt_exists(self, env_id: str) -> None:
        env = gym.make(env_id, max_turns=10)
        from glyphbench.core.base_env import BaseAsciiEnv
        unwrapped: BaseAsciiEnv = env.unwrapped  # type: ignore[assignment]
        prompt = unwrapped.system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50
