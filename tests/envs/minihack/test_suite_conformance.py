"""Suite-level conformance test for all registered MiniHack environments.

Iterates over every env matching glyphbench/minihack-* and validates the contract.
"""

from __future__ import annotations

from glyphbench.core import make_env, all_glyphbench_env_ids
import glyphbench.envs.minihack  # register envs


import pytest


def _minihack_env_ids() -> list[str]:
    """Collect all registered glyphbench/minihack-* env IDs."""
    return sorted(
        eid
        for eid in all_glyphbench_env_ids()
        if isinstance(eid, str) and eid.startswith("glyphbench/minihack-")
    )


@pytest.fixture(params=_minihack_env_ids())
def env_id(request: pytest.FixtureRequest) -> str:
    return request.param


class TestMiniHackSuiteConformance:
    def test_observation_contract(self, env_id: str) -> None:
        for seed in [0, 1, 2]:
            env = make_env(env_id, max_turns=50)
            obs, info = env.reset(seed)
            assert isinstance(obs, str)
            assert len(obs) > 0
            for _ in range(50):
                action = int(env.rng.integers(0, env.action_spec.n))
                obs, reward, terminated, truncated, info = env.step(action)
                assert isinstance(obs, str)
                assert len(obs) > 0
                if terminated or truncated:
                    break

    def test_action_space_is_22(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=10)
        from glyphbench.core.base_env import BaseGlyphEnv
        unwrapped: BaseGlyphEnv = env  # type: ignore[assignment]
        assert unwrapped.action_spec.n == 22

    def test_seed_reproducibility(self, env_id: str) -> None:
        for seed in [0, 42, 99]:
            e1 = make_env(env_id, max_turns=50)
            e2 = make_env(env_id, max_turns=50)
            o1, _ = e1.reset(seed)
            o2, _ = e2.reset(seed)
            assert o1 == o2
            for _ in range(10):
                a = 8  # WAIT (no-op)
                o1, r1, t1, tr1, _ = e1.step(a)
                o2, r2, t2, tr2, _ = e2.step(a)
                assert o1 == o2
                assert r1 == r2
                if t1 or tr1:
                    break

    def test_system_prompt_exists(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=10)
        from glyphbench.core.base_env import BaseGlyphEnv
        unwrapped: BaseGlyphEnv = env  # type: ignore[assignment]
        prompt = unwrapped.system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50
