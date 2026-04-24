"""Unit tests for Procgen BossFight env."""

from glyphbench.envs.procgen.bossfight import BossFightEnv


class TestBossFight:
    def _make(self, max_turns: int = 512) -> BossFightEnv:
        return BossFightEnv(max_turns=max_turns)

    def test_action_space(self):
        env = self._make()
        assert env.action_spec.n == 6
        assert "FIRE" in env.action_spec.names

    def test_env_id(self):
        assert self._make().env_id() == "glyphbench/procgen-bossfight-v0"

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

    def test_fire_creates_bullet(self):
        env = self._make()
        env.reset(0)
        fire = env.action_spec.index_of("FIRE")
        env.step(fire)
        bullets = [e for e in env._entities if e.etype == "bullet"]
        assert len(bullets) >= 1

    def test_boss_takes_damage(self):
        env = self._make()
        env.reset(0)
        # Place agent directly below boss
        env._agent_x = env._boss_x
        env._agent_y = env._boss_y + 1
        fire = env.action_spec.index_of("FIRE")
        # Fire upward -- bullet spawns at agent_y-1 = boss_y
        _, reward, _, _, info = env.step(fire)
        # The bullet is at boss_y, boss is at boss_y => hit
        assert reward >= 1.0
        assert info["boss_hp"] < 10

    def test_boss_phases(self):
        env = self._make()
        env.reset(0)
        assert env._boss_phase() == 1
        env._boss_hp = 5
        assert env._boss_phase() == 2
        env._boss_hp = 2
        assert env._boss_phase() == 3

    def test_boss_defeat(self):
        env = self._make()
        env.reset(0)
        env._boss_hp = 1
        # Place agent directly below boss
        env._agent_x = env._boss_x
        env._agent_y = env._boss_y + 1
        fire = env.action_spec.index_of("FIRE")
        _, reward, terminated, _, _ = env.step(fire)
        assert reward >= 10.0
        assert terminated

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
