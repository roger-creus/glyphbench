"""Unit tests for Procgen Leaper env."""

from atlas_rl.envs.procgen.leaper import LeaperEnv


class TestLeaper:
    def _make(self, max_turns: int = 512) -> LeaperEnv:
        return LeaperEnv(max_turns=max_turns)

    def test_action_space(self):
        env = self._make()
        assert env.action_spec.n == 5
        assert env.action_spec.names == ("NOOP", "LEFT", "RIGHT", "UP", "DOWN")

    def test_env_id(self):
        assert self._make().env_id() == "atlas_rl/procgen-leaper-v0"

    def test_reset_determinism(self):
        e1, e2 = self._make(), self._make()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    def test_step_determinism(self):
        e1, e2 = self._make(), self._make()
        e1.reset(seed=0)
        e2.reset(seed=0)
        up = e1.action_spec.index_of("UP")
        for _ in range(10):
            o1, r1, t1, tr1, _ = e1.step(up)
            o2, r2, t2, tr2, _ = e2.step(up)
            assert o1 == o2 and r1 == r2 and t1 == t2
            if t1 or tr1:
                break

    def test_observation_contract(self):
        env = self._make()
        obs, _ = env.reset(seed=0)
        assert "[Grid]" in obs
        assert "[Legend]" in obs

    def test_has_goal_row(self):
        env = self._make()
        env.reset(seed=0)
        # Top row should be goal
        assert env._lane_types[0] == "goal"

    def test_goal_reward(self):
        env = self._make()
        env.reset(seed=0)
        # Place agent on row 1, move up to goal row 0
        env._agent_x = env.GRID_W // 2
        env._agent_y = 1
        # Make sure row 1 is safe
        up = env.action_spec.index_of("UP")
        _, reward, terminated, _, _ = env.step(up)
        if env._agent_y == 0:
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

    def test_agent_starts_at_bottom(self):
        env = self._make()
        env.reset(seed=0)
        assert env._agent_y == env.GRID_H - 1

    def test_car_entities_exist(self):
        env = self._make()
        env.reset(seed=0)
        cars = [e for e in env._entities if e.etype == "car"]
        assert len(cars) > 0

    def test_system_prompt(self):
        env = self._make()
        prompt = env.system_prompt()
        assert len(prompt) > 0
