"""Unit tests for Atari Amidar env."""

import pytest

from glyphbench.envs.atari.amidar import AmidarEnv


class TestAmidar:
    """Tests for the Amidar environment."""

    def _make_env(self, max_turns: int = 10000) -> AmidarEnv:
        return AmidarEnv(max_turns=max_turns)

    def test_action_space_defined(self):
        env = self._make_env()
        assert env.action_spec.n == 5
        expected = ("NOOP", "UP", "RIGHT", "LEFT", "DOWN")
        assert env.action_spec.names == expected

    def test_env_id(self):
        env = self._make_env()
        assert env.env_id() == "glyphbench/atari-amidar-v0"

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
        right = e1.action_spec.index_of("RIGHT")
        actions = [right, right, noop, noop, noop]
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

    def test_painting_segments(self):
        env = self._make_env()
        env.reset(0)
        initial_painted = env._painted_segments
        # Move right (should paint unpainted segments)
        right = env.action_spec.index_of("RIGHT")
        total_reward = 0.0
        for _ in range(5):
            _, r, _, _, _ = env.step(right)
            total_reward += r
        assert env._painted_segments > initial_painted or total_reward > 0

    def test_scoring(self):
        env = self._make_env()
        env.reset(0)
        # Place unpainted segment next to player
        env._set_cell(env._player_x + 1, env._player_y, "\u00b7")
        right = env.action_spec.index_of("RIGHT")
        _, reward, _, _, _ = env.step(right)
        # Pattern D: +1/_WIN_TARGET per painted segment.
        assert reward >= 1.0 / env._WIN_TARGET
        assert env._score >= 1

    def test_enemy_collision_loses_life(self):
        env = self._make_env()
        env.reset(0)
        initial_lives = env._lives
        # Place enemy on player
        env._entities = [e for e in env._entities if e.etype != "enemy"]
        enemy = env._add_entity("enemy", "e", env._player_x, env._player_y)
        enemy.data["dir"] = (0, 0)
        enemy.data["patrol"] = "horizontal"
        noop = env.action_spec.index_of("NOOP")
        env.step(noop)
        assert env._lives < initial_lives

    def test_progress_tracking(self):
        env = self._make_env()
        env.reset(0)
        noop = env.action_spec.index_of("NOOP")
        _, _, _, _, info = env.step(noop)
        assert "painted" in info
        assert "total" in info
        assert "progress" in info

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
        assert "amidar" in prompt.lower() or "paint" in prompt.lower()
