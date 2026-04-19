"""Unit tests for Atari Wizard of Wor env."""

import pytest

from atlas_rl.envs.atari.wizardofwor import WizardOfWorEnv


class TestWizardOfWor:
    """Tests for the Wizard of Wor environment."""

    def _make_env(self, max_turns: int = 10000) -> WizardOfWorEnv:
        return WizardOfWorEnv(max_turns=max_turns)

    def test_action_space_defined(self):
        env = self._make_env()
        assert env.action_spec.n == 6
        expected = ("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN")
        assert env.action_spec.names == expected

    def test_env_id(self):
        env = self._make_env()
        assert env.env_id() == "atlas_rl/atari-wizardofwor-v0"

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
        obs_str, _ = env.reset(seed=0)
        assert isinstance(obs_str, str)
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str
        assert "[HUD]" in obs_str

    def test_grid_dimensions(self):
        env = self._make_env()
        env.reset(seed=0)
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        assert len(grid_lines) == 16
        assert all(len(line) == 20 for line in grid_lines)

    def test_shooting_kills_ghost(self):
        env = self._make_env()
        env.reset(seed=0)
        # Clear existing entities. Bullet spawns at player_x+1 and moves to
        # player_x+2 on the fire step. Collision is checked before ghosts
        # move, so place ghost at player_x+2.
        env._entities = [e for e in env._entities if e.etype not in ("ghost", "wizard")]
        from atlas_rl.envs.atari.wizardofwor import _GHOST_CHAR
        env._set_cell(env._player_x + 1, env._player_y, " ")
        env._set_cell(env._player_x + 2, env._player_y, " ")
        ghost = env._add_entity("ghost", _GHOST_CHAR, env._player_x + 2, env._player_y)
        ghost.data["dir"] = (1, 0)
        ghost.data["patrol_row"] = env._player_y
        env._total_enemies = 1
        env._facing = (1, 0)
        fire = env.action_spec.index_of("FIRE")
        _, reward, _, _, _ = env.step(fire)
        assert reward >= 100.0 or env._enemies_killed > 0

    def test_ghost_collision_loses_life(self):
        env = self._make_env()
        env.reset(seed=0)
        initial_lives = env._lives
        # Place ghost on player
        env._entities = [e for e in env._entities if e.etype not in ("ghost", "wizard")]
        from atlas_rl.envs.atari.wizardofwor import _GHOST_CHAR
        ghost = env._add_entity("ghost", _GHOST_CHAR, env._player_x, env._player_y)
        ghost.data["dir"] = (0, 0)
        ghost.data["patrol_row"] = env._player_y
        env._total_enemies = 1
        noop = env.action_spec.index_of("NOOP")
        env.step(noop)
        assert env._lives < initial_lives

    def test_rollout_no_crash(self):
        env = self._make_env(max_turns=200)
        env.reset(seed=42)
        for _ in range(200):
            action = int(env.rng.integers(0, env.action_spec.n))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    def test_max_turns_truncation(self):
        env = self._make_env(max_turns=5)
        env.reset(seed=0)
        noop = env.action_spec.index_of("NOOP")
        for i in range(5):
            _, _, terminated, truncated, _ = env.step(noop)
            if terminated:
                return
            if i == 4:
                assert truncated

    def test_reset_requires_seed(self):
        env = self._make_env()
        with pytest.raises(ValueError):
            env.reset()

    def test_system_prompt(self):
        env = self._make_env()
        prompt = env.system_prompt()
        assert len(prompt) > 0
        assert "wizard" in prompt.lower() or "ghost" in prompt.lower()
