"""Unit tests for Atari Freeway env."""

from glyphbench.envs.atari.freeway import FreewayEnv


class TestFreeway:
    """Tests for the Freeway environment."""

    def _make_env(self, max_turns: int = 10000) -> FreewayEnv:
        return FreewayEnv(max_turns=max_turns)

    def test_action_space_defined(self) -> None:
        env = self._make_env()
        assert env.action_spec.n == 3
        assert "NOOP" in env.action_spec.names
        assert "UP" in env.action_spec.names
        assert "DOWN" in env.action_spec.names

    def test_noop_action_name(self) -> None:
        env = self._make_env()
        assert env.noop_action_name == "NOOP"

    def test_env_id(self) -> None:
        env = self._make_env()
        assert env.env_id() == "glyphbench/atari-freeway-v0"

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
        up = e1.action_spec.index_of("UP")
        actions = [noop, up, up, noop, up]
        for a in actions:
            o1, r1, t1, tr1, _ = e1.step(a)
            o2, r2, t2, tr2, _ = e2.step(a)
            assert o1 == o2
            assert r1 == r2
            assert t1 == t2

    def test_observation_contract(self) -> None:
        env = self._make_env()
        obs_str, _ = env.reset(0)
        assert isinstance(obs_str, str)
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str
        assert "[HUD]" in obs_str

    def test_grid_dimensions(self) -> None:
        env = self._make_env()
        env.reset(0)
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        assert len(grid_lines) == 16
        lengths = [len(line) for line in grid_lines]
        assert len(set(lengths)) == 1

    def test_starts_at_bottom(self) -> None:
        env = self._make_env()
        env.reset(0)
        assert env._player_y == env._PLAYER_START_Y

    def test_lanes_created(self) -> None:
        env = self._make_env()
        env.reset(0)
        assert len(env._lanes) > 0

    def test_up_moves_player(self) -> None:
        env = self._make_env()
        env.reset(0)
        start_y = env._player_y
        up = env.action_spec.index_of("UP")
        env.step(up)
        assert env._player_y == start_y - 1

    def test_never_terminates(self) -> None:
        """Freeway never terminates, only truncates."""
        env = self._make_env(max_turns=50)
        env.reset(42)
        noop = env.action_spec.index_of("NOOP")
        terminated_ever = False
        for _ in range(50):
            _, _, terminated, truncated, _ = env.step(noop)
            if terminated:
                terminated_ever = True
            if truncated:
                break
        assert not terminated_ever

    def test_rollout_no_crash(self) -> None:
        env = self._make_env(max_turns=200)
        env.reset(42)
        actions = list(range(env.action_spec.n))
        for i in range(200):
            a = actions[i % len(actions)]
            _, _, terminated, truncated, _ = env.step(a)
            if terminated or truncated:
                break

    def test_max_turns_truncation(self) -> None:
        env = self._make_env(max_turns=5)
        env.reset(0)
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
        with pytest.raises(TypeError):
            env.reset()
