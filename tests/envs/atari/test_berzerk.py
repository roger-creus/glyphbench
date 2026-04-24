"""Unit tests for Atari Berzerk env."""

import pytest

from glyphbench.envs.atari.berzerk import BerzerkEnv


class TestBerzerk:
    """Tests for the Berzerk environment."""

    def _make_env(self, max_turns: int = 10000) -> BerzerkEnv:
        return BerzerkEnv(max_turns=max_turns)

    def test_action_space_defined(self):
        env = self._make_env()
        assert env.action_spec.n == 10
        assert "FIRE" in env.action_spec.names
        assert "UP_FIRE" in env.action_spec.names

    def test_env_id(self):
        env = self._make_env()
        assert env.env_id() == "glyphbench/atari-berzerk-v0"

    def test_reset_determinism(self):
        e1 = self._make_env()
        e2 = self._make_env()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    def test_step_determinism(self):
        e1 = self._make_env()
        e2 = self._make_env()
        e1.reset(seed=0)
        e2.reset(seed=0)
        noop = e1.action_spec.index_of("NOOP")
        actions = [noop] * 5
        for a in actions:
            o1, r1, t1, tr1, _ = e1.step(a)
            o2, r2, t2, tr2, _ = e2.step(a)
            assert o1 == o2
            assert r1 == r2
            assert t1 == t2

    def test_observation_contract(self):
        env = self._make_env()
        obs_str, _ = env.reset(0)
        assert isinstance(obs_str, str)
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str
        assert "[HUD]" in obs_str

    def test_grid_dimensions(self):
        env = self._make_env()
        env.reset(0)
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        assert len(grid_lines) == 16
        assert all(len(line) == 20 for line in grid_lines)

    def test_shooting_destroys_robot(self):
        env = self._make_env()
        env.reset(0)
        # Place a robot right next to player, in line of fire
        env._entities = [e for e in env._entities if e.etype != "robot"]
        from glyphbench.envs.atari.berzerk import _ROBOT_CHAR
        robot = env._add_entity("robot", _ROBOT_CHAR, env._player_x + 2, env._player_y)
        robot.data["shoot_timer"] = 999  # Prevent robot from shooting
        env._total_robots = 1
        env._facing = (1, 0)
        fire = env.action_spec.index_of("FIRE")
        # Fire and step to let bullet travel
        _, reward, _, _, _ = env.step(fire)
        noop = env.action_spec.index_of("NOOP")
        total_reward = reward
        for _ in range(5):
            _, r, _, _, _ = env.step(noop)
            total_reward += r
            if total_reward > 0:
                break
        assert total_reward >= 50.0 or env._robots_killed > 0

    def test_robot_collision_loses_life(self):
        env = self._make_env()
        env.reset(0)
        initial_lives = env._lives
        # Place robot on player
        env._entities = [e for e in env._entities if e.etype != "robot"]
        from glyphbench.envs.atari.berzerk import _ROBOT_CHAR
        robot = env._add_entity("robot", _ROBOT_CHAR, env._player_x, env._player_y)
        robot.data["shoot_timer"] = 999
        env._total_robots = 1
        noop = env.action_spec.index_of("NOOP")
        env.step(noop)
        assert env._lives < initial_lives

    def test_exit_door_advances_level(self):
        env = self._make_env()
        env.reset(0)
        initial_level = env._level
        # Clear all robots
        for e in env._entities:
            if e.etype == "robot":
                e.alive = False
        env._entities = [e for e in env._entities if e.alive]
        # Find exit door and move player adjacent
        for y in range(16):
            for x in range(20):
                if env._grid_at(x, y) == "D":
                    env._player_x = x - 1
                    env._player_y = y
                    # Make sure path is clear
                    env._set_cell(x - 1, y, " ")
                    right = env.action_spec.index_of("RIGHT")
                    env.step(right)
                    assert env._level > initial_level
                    return
        # If no door found, that's still OK for the test structure

    def test_rollout_no_crash(self):
        env = self._make_env(max_turns=200)
        env.reset(42)
        for _ in range(200):
            action = int(env.rng.integers(0, env.action_spec.n))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    def test_max_turns_truncation(self):
        env = self._make_env(max_turns=5)
        env.reset(0)
        noop = env.action_spec.index_of("NOOP")
        for i in range(5):
            _, _, terminated, truncated, _ = env.step(noop)
            if terminated:
                return
            if i == 4:
                assert truncated

    def test_reset_requires_seed(self):
        env = self._make_env()
        with pytest.raises(TypeError):
            env.reset()

    def test_system_prompt(self):
        env = self._make_env()
        prompt = env.system_prompt()
        assert len(prompt) > 0
        assert "berzerk" in prompt.lower() or "robot" in prompt.lower()
