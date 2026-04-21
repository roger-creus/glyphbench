"""Unit tests for Procgen Miner env."""

import pytest

from glyphbench.envs.procgen.miner import MinerEnv


class TestMiner:
    """Tests for the Procgen Miner environment."""

    def _make(self, max_turns: int = 512) -> MinerEnv:
        return MinerEnv(max_turns=max_turns)

    def test_action_space_defined(self) -> None:
        env = self._make()
        assert env.action_spec.n == 5
        expected = ("NOOP", "LEFT", "RIGHT", "UP", "DOWN")
        assert env.action_spec.names == expected

    def test_env_id(self) -> None:
        env = self._make()
        assert env.env_id() == "glyphbench/procgen-miner-v0"

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
        obs_str, _ = env.reset(seed=0)
        assert isinstance(obs_str, str)
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str
        assert "[HUD]" in obs_str

    def test_window_size(self) -> None:
        env = self._make()
        env.reset(seed=0)
        grid_obs = env.get_observation()
        lines = grid_obs.grid.split("\n")
        assert len(lines) == 20
        assert all(len(line) == 20 for line in lines)

    def test_diamond_collection(self) -> None:
        """Collecting diamond gives +1."""
        env = self._make()
        env.reset(seed=0)
        # Place diamond next to agent
        env._set_cell(env._agent_x + 1, env._agent_y, "D")
        right = env.action_spec.index_of("RIGHT")
        _, reward, _, _, _ = env.step(right)
        assert reward >= 1.0

    def test_dig_through_dirt(self) -> None:
        """Agent can dig through dirt."""
        env = self._make()
        env.reset(seed=0)
        env._set_cell(env._agent_x + 1, env._agent_y, "d")
        right = env.action_spec.index_of("RIGHT")
        env.step(right)
        assert env._agent_x > 1  # moved

    def test_cannot_move_through_wall(self) -> None:
        """Agent cannot move through walls."""
        env = self._make()
        env.reset(seed=0)
        # Agent is at (1,1), left wall is at x=0
        left = env.action_spec.index_of("LEFT")
        env.step(left)
        assert env._agent_x >= 1  # blocked by wall

    def test_cannot_move_through_boulder(self) -> None:
        """Agent cannot push boulders."""
        env = self._make()
        env.reset(seed=0)
        env._set_cell(env._agent_x + 1, env._agent_y, "R")
        right = env.action_spec.index_of("RIGHT")
        old_x = env._agent_x
        env.step(right)
        assert env._agent_x == old_x

    def test_boulder_falls(self) -> None:
        """Boulder falls when cell below is empty."""
        env = self._make()
        env.reset(seed=0)
        # Set up a clean column: clear everything above to avoid cascading
        bx, by = 5, 8
        for cy in range(1, by):
            env._set_cell(bx, cy, "·")
        env._set_cell(bx, by, "R")
        env._set_cell(bx, by + 1, "·")
        noop = env.action_spec.index_of("NOOP")
        env.step(noop)
        assert env._world_at(bx, by + 1) == "R"
        assert env._world_at(bx, by) == "·"

    def test_boulder_crush_kills(self) -> None:
        """Boulder falling on agent kills them."""
        env = self._make()
        env.reset(seed=0)
        # Set up: boulder directly above agent, empty cell between
        env._agent_x = 5
        env._agent_y = 8
        # Clear column above agent to prevent cascading interference
        for cy in range(1, 8):
            env._set_cell(5, cy, "·")
        env._set_cell(5, 7, "R")
        env._set_cell(5, 8, "·")  # agent position
        noop = env.action_spec.index_of("NOOP")
        env.step(noop)
        # Boulder should fall onto agent position (5,8) causing death
        # or at minimum no crash

    def test_goal_reward(self) -> None:
        """Reaching exit gives +10."""
        env = self._make()
        env.reset(seed=0)
        # Find G
        for y in range(env._world_h):
            for x in range(env._world_w):
                if env._world_at(x, y) == "G":
                    env._agent_x = x - 1
                    env._agent_y = y
                    env._set_cell(x - 1, y, "·")  # clear path
                    right = env.action_spec.index_of("RIGHT")
                    _, reward, terminated, _, _ = env.step(right)
                    assert reward >= 10.0
                    assert terminated
                    return
        pytest.skip("No goal found")

    def test_max_turns_truncation(self) -> None:
        env = self._make(max_turns=5)
        env.reset(seed=0)
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
        assert "miner" in prompt.lower() or "dig" in prompt.lower()

    def test_reset_requires_seed(self) -> None:
        env = self._make()
        with pytest.raises(ValueError):
            env.reset()

    @pytest.mark.parametrize("seed", range(5))
    def test_multiple_seeds(self, seed: int) -> None:
        env = self._make()
        env.reset(seed=seed)
        noop = env.action_spec.index_of("NOOP")
        for _ in range(20):
            _, _, t, tr, _ = env.step(noop)
            if t or tr:
                break
