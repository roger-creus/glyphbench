"""Unit tests for Procgen Chaser env."""

from atlas_rl.envs.procgen.chaser import ChaserEnv


class TestChaser:
    def _make(self, max_turns: int = 512) -> ChaserEnv:
        return ChaserEnv(max_turns=max_turns)

    def test_action_space(self):
        env = self._make()
        assert env.action_spec.n == 5
        assert env.action_spec.names == ("NOOP", "LEFT", "RIGHT", "UP", "DOWN")

    def test_env_id(self):
        assert self._make().env_id() == "atlas_rl/procgen-chaser-v0"

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

    def test_has_pellets(self):
        env = self._make()
        env.reset(seed=0)
        assert env._pellet_count > 0

    def test_has_ghosts(self):
        env = self._make()
        env.reset(seed=0)
        ghosts = [e for e in env._entities if e.etype == "ghost"]
        assert len(ghosts) >= 2

    def test_pellet_collection(self):
        env = self._make()
        env.reset(seed=0)
        initial = env._pellet_count
        # Move around to collect pellets
        right = env.action_spec.index_of("RIGHT")
        total_reward = 0.0
        for _ in range(20):
            _, reward, terminated, _, _ = env.step(right)
            total_reward += reward
            if terminated:
                break
        # Should have collected at least some pellets or been caught
        assert total_reward > 0 or env._pellet_count < initial or terminated

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

    def test_power_pellet_scares_ghosts(self):
        env = self._make()
        env.reset(seed=0)
        # Find power pellet and place agent on it
        for y in range(env.MAZE_H):
            for x in range(env.MAZE_W):
                if env._world_at(x, y) == "O":
                    env._agent_x = x
                    env._agent_y = y
                    noop = env.action_spec.index_of("NOOP")
                    env.step(noop)
                    assert env._power_timer > 0
                    for e in env._entities:
                        if e.alive and e.etype == "ghost":
                            assert e.data["scared"]
                    return

    def test_system_prompt(self):
        env = self._make()
        prompt = env.system_prompt()
        assert len(prompt) > 0
