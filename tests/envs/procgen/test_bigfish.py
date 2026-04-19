"""Unit tests for Procgen BigFish env."""

from atlas_rl.envs.procgen.bigfish import BigFishEnv


class TestBigFish:
    def _make(self, max_turns: int = 512) -> BigFishEnv:
        return BigFishEnv(max_turns=max_turns)

    def test_action_space(self):
        env = self._make()
        assert env.action_spec.n == 5
        assert env.action_spec.names == ("NOOP", "LEFT", "RIGHT", "UP", "DOWN")

    def test_env_id(self):
        assert self._make().env_id() == "atlas_rl/procgen-bigfish-v0"

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

    def test_has_fish(self):
        env = self._make()
        env.reset(seed=0)
        fish = [e for e in env._entities if e.etype == "fish"]
        assert len(fish) > 0

    def test_eat_smaller_fish(self):
        env = self._make()
        env.reset(seed=0)
        # Set agent size to 3 and place a small fish on agent
        env._agent_size = 3
        # Add a size-1 fish at agent position
        env._add_entity("fish", "f", env._agent_x, env._agent_y, dx=0, data={"size": 1})
        noop = env.action_spec.index_of("NOOP")
        _, reward, terminated, _, _ = env.step(noop)
        assert reward >= 1.0
        assert not terminated

    def test_eaten_by_bigger_fish(self):
        env = self._make()
        env.reset(seed=0)
        env._agent_size = 1
        # Add a big fish at agent position
        env._add_entity("fish", "W", env._agent_x, env._agent_y, dx=0, data={"size": 6})
        noop = env.action_spec.index_of("NOOP")
        _, reward, terminated, _, _ = env.step(noop)
        assert reward == -1.0
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

    def test_agent_grows(self):
        env = self._make()
        env.reset(seed=0)
        env._agent_size = 1
        env._fish_eaten = 2  # next eat will trigger growth
        # Place a small fish on agent
        env._add_entity("fish", "f", env._agent_x, env._agent_y, dx=0, data={"size": 1})
        noop = env.action_spec.index_of("NOOP")
        env.step(noop)
        assert env._agent_size == 2

    def test_system_prompt(self):
        env = self._make()
        prompt = env.system_prompt()
        assert len(prompt) > 0
