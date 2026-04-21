"""Unit tests for Procgen Plunder env."""

from glyphbench.envs.procgen.plunder import PlunderEnv


class TestPlunder:
    def _make(self, max_turns: int = 512) -> PlunderEnv:
        return PlunderEnv(max_turns=max_turns)

    def test_action_space(self):
        env = self._make()
        assert env.action_spec.n == 5
        assert "FIRE" in env.action_spec.names

    def test_env_id(self):
        assert self._make().env_id() == "glyphbench/procgen-plunder-v0"

    def test_reset_determinism(self):
        e1, e2 = self._make(), self._make()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    def test_step_determinism(self):
        e1, e2 = self._make(), self._make()
        e1.reset(seed=0)
        e2.reset(seed=0)
        noop = e1.action_spec.index_of("NOOP")
        for _ in range(10):
            o1, r1, t1, tr1, _ = e1.step(noop)
            o2, r2, t2, tr2, _ = e2.step(noop)
            assert o1 == o2 and r1 == r2 and t1 == t2
            if t1 or tr1:
                break

    def test_observation_contract(self):
        env = self._make()
        obs, _ = env.reset(seed=0)
        assert "[Grid]" in obs
        assert "[Legend]" in obs

    def test_fire_creates_cannonball(self):
        env = self._make()
        env.reset(seed=0)
        fire = env.action_spec.index_of("FIRE")
        env.step(fire)
        balls = [e for e in env._entities if e.etype == "cannonball"]
        assert len(balls) >= 1

    def test_hit_pirate(self):
        env = self._make()
        env.reset(seed=0)
        # Place pirate right above agent
        env._entities = []
        env._add_entity("pirate", "P", env._agent_x, env._agent_y - 1, dx=0, dy=0)
        fire = env.action_spec.index_of("FIRE")
        _, reward, _, _, _ = env.step(fire)
        assert reward >= 1.0

    def test_hit_civilian_penalty(self):
        env = self._make()
        env.reset(seed=0)
        # Place civilian right above agent
        env._entities = []
        env._add_entity("civilian", "c", env._agent_x, env._agent_y - 1, dx=0, dy=0)
        fire = env.action_spec.index_of("FIRE")
        _, reward, _, _, _ = env.step(fire)
        assert reward <= -1.0

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

    def test_system_prompt(self):
        env = self._make()
        prompt = env.system_prompt()
        assert len(prompt) > 0
