"""Unit tests for Atari Space Invaders env."""

from glyphbench.envs.atari.spaceinvaders import SpaceInvadersEnv


class TestSpaceInvaders:
    """Tests for the Space Invaders environment."""

    def _make_env(self, max_turns: int = 10000) -> SpaceInvadersEnv:
        return SpaceInvadersEnv(max_turns=max_turns)

    def test_action_space_defined(self) -> None:
        env = self._make_env()
        assert env.action_spec.n == 6
        assert "NOOP" in env.action_spec.names
        assert "FIRE" in env.action_spec.names
        assert "LEFT_FIRE" in env.action_spec.names

    def test_noop_action_name(self) -> None:
        env = self._make_env()
        assert env.noop_action_name == "NOOP"

    def test_env_id(self) -> None:
        env = self._make_env()
        assert env.env_id() == "glyphbench/atari-spaceinvaders-v0"

    def test_reset_determinism(self) -> None:
        e1 = self._make_env()
        e2 = self._make_env()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    def test_step_determinism(self) -> None:
        e1 = self._make_env()
        e2 = self._make_env()
        e1.reset(seed=0)
        e2.reset(seed=0)
        noop = e1.action_spec.index_of("NOOP")
        fire = e1.action_spec.index_of("FIRE")
        actions = [noop, fire, noop, noop, fire]
        for a in actions:
            o1, r1, t1, tr1, _ = e1.step(a)
            o2, r2, t2, tr2, _ = e2.step(a)
            assert o1 == o2
            assert r1 == r2
            assert t1 == t2

    def test_observation_contract(self) -> None:
        env = self._make_env()
        obs_str, _ = env.reset(seed=0)
        assert isinstance(obs_str, str)
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str
        assert "[HUD]" in obs_str

    def test_grid_dimensions(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        assert len(grid_lines) == 20
        lengths = [len(line) for line in grid_lines]
        assert len(set(lengths)) == 1

    def test_aliens_created(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        alive = sum(
            1 for row in env._aliens for a in row if a is not None and a.alive
        )
        assert alive > 0

    def test_shields_created(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        assert len(env._shields) > 0

    def test_lives_start_at_3(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        assert env._lives == 3

    def test_rollout_no_crash(self) -> None:
        env = self._make_env(max_turns=200)
        env.reset(seed=42)
        actions = list(range(env.action_spec.n))
        for i in range(200):
            a = actions[i % len(actions)]
            _, _, terminated, truncated, _ = env.step(a)
            if terminated or truncated:
                break

    def test_max_turns_truncation(self) -> None:
        env = self._make_env(max_turns=5)
        env.reset(seed=0)
        noop = env.action_spec.index_of("NOOP")
        for i in range(5):
            _, _, terminated, truncated, _ = env.step(noop)
            if terminated:
                return
            if i == 4:
                assert truncated

    def test_system_prompt(self) -> None:
        env = self._make_env()
        prompt = env.system_prompt()
        assert len(prompt) > 0

    def test_reset_requires_seed(self) -> None:
        import pytest

        env = self._make_env()
        with pytest.raises(ValueError):
            env.reset()
