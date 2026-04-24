"""Unit tests for Procgen FruitBot env."""

import pytest

from glyphbench.envs.procgen.fruitbot import FruitBotEnv


class TestFruitBot:
    """Tests for the Procgen FruitBot environment."""

    def _make(self, max_turns: int = 512) -> FruitBotEnv:
        return FruitBotEnv(max_turns=max_turns)

    def test_action_space_defined(self) -> None:
        env = self._make()
        assert env.action_spec.n == 4
        expected = ("NOOP", "LEFT", "RIGHT", "DOWN")
        assert env.action_spec.names == expected

    def test_env_id(self) -> None:
        env = self._make()
        assert env.env_id() == "glyphbench/procgen-fruitbot-v0"

    def test_reset_determinism(self) -> None:
        e1, e2 = self._make(), self._make()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    def test_step_determinism(self) -> None:
        e1, e2 = self._make(), self._make()
        e1.reset(seed=0)
        e2.reset(seed=0)
        noop = e1.action_spec.index_of("NOOP")
        for _ in range(10):
            o1, r1, t1, tr1, _ = e1.step(noop)
            o2, r2, t2, tr2, _ = e2.step(noop)
            assert o1 == o2
            assert r1 == r2
            assert t1 == t2
            if t1 or tr1:
                break

    def test_observation_contract(self) -> None:
        env = self._make()
        obs_str, _ = env.reset(0)
        assert isinstance(obs_str, str)
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str
        assert "[HUD]" in obs_str

    def test_window_size(self) -> None:
        env = self._make()
        env.reset(0)
        grid_obs = env.get_observation()
        lines = grid_obs.grid.split("\n")
        assert len(lines) == 20
        assert all(len(line) == 14 for line in lines)

    def test_fruit_collection_reward(self) -> None:
        """Collecting fruit gives +1."""
        env = self._make()
        env.reset(0)
        # Place fruit below agent
        env._set_cell(env._agent_x, env._agent_y + 1, "%")
        noop = env.action_spec.index_of("NOOP")
        _, reward, _, _, _ = env.step(noop)
        assert reward >= 1.0

    def test_obstacle_penalty(self) -> None:
        """Hitting obstacle gives -1."""
        env = self._make()
        env.reset(0)
        # Place obstacle below agent
        env._set_cell(env._agent_x, env._agent_y + 1, "x")
        noop = env.action_spec.index_of("NOOP")
        _, reward, _, _, _ = env.step(noop)
        assert reward <= -1.0

    def test_agent_falls(self) -> None:
        """Agent falls each step."""
        env = self._make()
        env.reset(0)
        initial_y = env._agent_y
        noop = env.action_spec.index_of("NOOP")
        env.step(noop)
        assert env._agent_y > initial_y

    def test_down_falls_faster(self) -> None:
        """DOWN action falls 2 cells instead of 1."""
        env = self._make()
        env.reset(0)
        initial_y = env._agent_y
        # Clear the cells below
        env._set_cell(env._agent_x, initial_y + 1, ".")
        env._set_cell(env._agent_x, initial_y + 2, ".")
        down = env.action_spec.index_of("DOWN")
        env.step(down)
        assert env._agent_y >= initial_y + 2

    def test_terminates_at_bottom(self) -> None:
        """Game terminates when agent reaches the bottom."""
        env = self._make()
        env.reset(0)
        noop = env.action_spec.index_of("NOOP")
        terminated = False
        for _ in range(200):
            _, _, terminated, truncated, _ = env.step(noop)
            if terminated or truncated:
                break
        assert terminated

    def test_max_turns_truncation(self) -> None:
        env = self._make(max_turns=5)
        env.reset(0)
        noop = env.action_spec.index_of("NOOP")
        for i in range(5):
            _, _, terminated, truncated, _ = env.step(noop)
            if terminated:
                return
            if i == 4:
                assert truncated

    def test_system_prompt(self) -> None:
        env = self._make()
        prompt = env.system_prompt()
        assert len(prompt) > 0
        assert "fruit" in prompt.lower() or "FruitBot" in prompt

    def test_reset_requires_seed(self) -> None:
        env = self._make()
        with pytest.raises(TypeError):
            env.reset()

    @pytest.mark.parametrize("seed", range(5))
    def test_multiple_seeds(self, seed: int) -> None:
        env = self._make()
        env.reset(seed=seed)
        noop = env.action_spec.index_of("NOOP")
        for _ in range(50):
            _, _, t, tr, _ = env.step(noop)
            if t or tr:
                break
