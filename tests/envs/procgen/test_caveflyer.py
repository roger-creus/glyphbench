"""Unit tests for Procgen CaveFlyer env."""

from glyphbench.envs.procgen.caveflyer import CaveFlyerEnv


class TestCaveFlyer:
    def _make(self, max_turns: int = 512) -> CaveFlyerEnv:
        return CaveFlyerEnv(max_turns=max_turns)

    def test_action_space(self):
        env = self._make()
        assert env.action_spec.n == 6
        assert "FIRE" in env.action_spec.names

    def test_env_id(self):
        assert self._make().env_id() == "glyphbench/procgen-caveflyer-v0"

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
        obs, _ = env.reset(0)
        assert "[Grid]" in obs
        assert "[Legend]" in obs

    def test_has_enemies(self):
        env = self._make()
        env.reset(0)
        enemies = [e for e in env._entities if e.etype == "enemy"]
        assert len(enemies) > 0

    def test_fire_creates_bullet(self):
        env = self._make()
        env.reset(0)
        fire = env.action_spec.index_of("FIRE")
        env.step(fire)
        bullets = [e for e in env._entities if e.etype == "bullet"]
        assert len(bullets) >= 1

    def test_bullet_hits_enemy(self):
        env = self._make()
        env.reset(0)
        # Clear entities and place enemy right of agent
        env._entities = []
        env._add_entity("enemy", "E", env._agent_x + 1, env._agent_y, dx=0, dy=0)
        fire = env.action_spec.index_of("FIRE")
        _, reward, _, _, _ = env.step(fire)
        assert reward > 0

    def test_random_rollout(self):
        env = self._make(max_turns=200)
        env.reset(7)
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
