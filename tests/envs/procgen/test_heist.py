"""Unit tests for Procgen Heist env."""

from glyphbench.envs.procgen.heist import HeistEnv


class TestHeist:
    def _make(self, max_turns: int = 512) -> HeistEnv:
        return HeistEnv(max_turns=max_turns)

    def test_action_space(self):
        env = self._make()
        assert env.action_spec.n == 5
        assert env.action_spec.names == ("NOOP", "LEFT", "RIGHT", "UP", "DOWN")

    def test_env_id(self):
        assert self._make().env_id() == "glyphbench/procgen-heist-v0"

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

    def test_has_goal(self):
        env = self._make()
        env.reset(seed=0)
        # Goal G should exist in the world
        found = False
        for y in range(env.MAZE_H):
            for x in range(env.MAZE_W):
                if env._world_at(x, y) == "G":
                    found = True
                    break
        assert found

    def test_key_pickup(self):
        env = self._make()
        env.reset(seed=0)
        # Find a key and place agent next to it
        for y in range(env.MAZE_H):
            for x in range(env.MAZE_W):
                if env._world_at(x, y) in ("r", "b", "y"):
                    # Place agent on the key cell
                    env._agent_x = x
                    env._agent_y = y
                    noop = env.action_spec.index_of("NOOP")
                    env.step(noop)
                    assert env._world_at(x, y) == "·"
                    return
        # If no key found, that's fine (rare seed)

    def test_goal_reward(self):
        env = self._make()
        env.reset(seed=0)
        # Give all keys and place agent next to goal
        env._keys_held = {"r", "b", "y"}
        env._agent_x = env._goal_x
        env._agent_y = env._goal_y
        noop = env.action_spec.index_of("NOOP")
        _, reward, terminated, _, _ = env.step(noop)
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

    def test_door_blocks_without_key(self):
        env = self._make()
        env.reset(seed=0)
        # Find a door
        for y in range(env.MAZE_H):
            for x in range(env.MAZE_W):
                if env._world_at(x, y) in ("R", "B", "Y"):
                    # Door should be solid without key
                    assert env._is_solid(x, y)
                    return

    def test_system_prompt(self):
        env = self._make()
        prompt = env.system_prompt()
        assert len(prompt) > 0
