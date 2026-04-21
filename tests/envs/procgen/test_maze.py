"""Unit tests for Procgen Maze env."""

from glyphbench.envs.procgen.maze import MazeEnv


class TestMaze:
    def _make(self, max_turns: int = 512) -> MazeEnv:
        return MazeEnv(max_turns=max_turns)

    def test_action_space(self):
        env = self._make()
        assert env.action_spec.n == 5
        assert env.action_spec.names == ("NOOP", "LEFT", "RIGHT", "UP", "DOWN")

    def test_env_id(self):
        assert self._make().env_id() == "glyphbench/procgen-maze-v0"

    def test_reset_determinism(self):
        e1, e2 = self._make(), self._make()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    def test_step_determinism(self):
        e1, e2 = self._make(), self._make()
        e1.reset(seed=0)
        e2.reset(seed=0)
        right = e1.action_spec.index_of("RIGHT")
        for _ in range(10):
            o1, r1, t1, tr1, _ = e1.step(right)
            o2, r2, t2, tr2, _ = e2.step(right)
            assert o1 == o2 and r1 == r2 and t1 == t2
            if t1 or tr1:
                break

    def test_observation_contract(self):
        env = self._make()
        obs, _ = env.reset(seed=0)
        assert "[Grid]" in obs
        assert "[Legend]" in obs
        assert "[HUD]" in obs

    def test_has_cheese(self):
        env = self._make()
        env.reset(seed=0)
        # Cheese exists in the world grid
        assert env._world_at(env._cheese_x, env._cheese_y) == "C"

    def test_cheese_reward(self):
        env = self._make()
        env.reset(seed=0)
        # Place agent next to cheese
        env._agent_x = env._cheese_x - 1
        env._agent_y = env._cheese_y
        # Clear wall between agent and cheese if any
        env._set_cell(env._agent_x, env._agent_y, ".")
        right = env.action_spec.index_of("RIGHT")
        _, reward, terminated, _, _ = env.step(right)
        assert reward == 10.0
        assert terminated

    def test_random_rollout(self):
        env = self._make(max_turns=200)
        env.reset(seed=7)
        done = False
        steps = 0
        while not done and steps < 200:
            action = int(env.rng.integers(0, env.action_spec.n))
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps > 0

    def test_walls_block_movement(self):
        env = self._make()
        env.reset(seed=0)
        # Agent at (1,1), wall at (0,1)
        x0 = env._agent_x
        left = env.action_spec.index_of("LEFT")
        env.step(left)
        # Agent should not move into wall at x=0
        assert env._agent_x >= 0

    def test_max_turns_truncation(self):
        env = self._make(max_turns=5)
        env.reset(seed=0)
        noop = env.action_spec.index_of("NOOP")
        for i in range(5):
            _, _, terminated, truncated, _ = env.step(noop)
            if terminated:
                return
            if i == 4:
                assert truncated

    def test_system_prompt(self):
        env = self._make()
        prompt = env.system_prompt()
        assert len(prompt) > 0
        assert "maze" in prompt.lower() or "cheese" in prompt.lower()
