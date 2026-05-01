"""Unit tests for Procgen Climber env."""

import pytest

from glyphbench.envs.procgen.climber import ClimberEnv


class TestClimber:
    """Tests for the Procgen Climber environment."""

    def _make(self, max_turns: int = 512) -> ClimberEnv:
        return ClimberEnv(max_turns=max_turns)

    def test_action_space_defined(self) -> None:
        env = self._make()
        assert env.action_spec.n == 6
        expected = ("NOOP", "LEFT", "RIGHT", "JUMP", "JUMP_LEFT", "JUMP_RIGHT")
        assert env.action_spec.names == expected

    def test_env_id(self) -> None:
        env = self._make()
        assert env.env_id() == "glyphbench/procgen-climber-v0"

    def test_reset_determinism(self) -> None:
        e1, e2 = self._make(), self._make()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    def test_step_determinism(self) -> None:
        e1, e2 = self._make(), self._make()
        e1.reset(seed=0)
        e2.reset(seed=0)
        right = e1.action_spec.index_of("RIGHT")
        for _ in range(10):
            o1, r1, t1, tr1, _ = e1.step(right)
            o2, r2, t2, tr2, _ = e2.step(right)
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

    def test_star_collection(self) -> None:
        """Stars give partial progress reward (Pattern B)."""
        env = self._make()
        env.reset(0)
        # Place a star next to agent
        env._set_cell(env._agent_x + 1, env._agent_y, "*")
        right = env.action_spec.index_of("RIGHT")
        _, reward, _, _, _ = env.step(right)
        assert reward > 0

    def test_goal_reward(self) -> None:
        """Reaching goal gives the milestone bonus and terminates."""
        env = self._make()
        env.reset(0)
        # Place agent next to goal
        # Find G
        for y in range(env._world_h):
            for x in range(env._world_w):
                if env._world_at(x, y) == "G":
                    env._agent_x = x - 1
                    env._agent_y = y
                    env._on_ground = True
                    env._jump_step = -1
                    right = env.action_spec.index_of("RIGHT")
                    _, reward, terminated, _, _ = env.step(right)
                    assert reward > 0
                    assert terminated
                    return
        pytest.skip("No goal found")

    def test_enemy_kills(self) -> None:
        """Touching an enemy terminates."""
        env = self._make()
        env.reset(0)
        if env._entities:
            e = env._entities[0]
            env._agent_x = e.x - 1
            env._agent_y = e.y
            env._on_ground = True
            env._jump_step = -1
            right = env.action_spec.index_of("RIGHT")
            _, _, terminated, _, info = env.step(right)
            if env._agent_x == e.x and env._agent_y == e.y:
                assert terminated
                return
        pytest.skip("No enemies in this seed")

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
        assert "climb" in prompt.lower() or "Climber" in prompt

    def test_reset_requires_seed(self) -> None:
        env = self._make()
        with pytest.raises(TypeError):
            env.reset()

    @pytest.mark.parametrize("seed", range(5))
    def test_multiple_seeds(self, seed: int) -> None:
        """Env runs without error across multiple seeds."""
        env = self._make()
        env.reset(seed=seed)
        noop = env.action_spec.index_of("NOOP")
        for _ in range(20):
            _, _, t, tr, _ = env.step(noop)
            if t or tr:
                break
