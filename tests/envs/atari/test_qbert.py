"""Unit tests for Atari Q*bert env."""

from glyphbench.envs.atari.qbert import QbertEnv


class TestQbert:
    """Tests for the Q*bert environment."""

    def _make_env(self, max_turns: int = 10000) -> QbertEnv:
        return QbertEnv(max_turns=max_turns)

    def test_action_space_defined(self) -> None:
        env = self._make_env()
        assert env.action_spec.n == 5
        assert "NOOP" in env.action_spec.names
        assert "UP_RIGHT" in env.action_spec.names
        assert "DOWN_LEFT" in env.action_spec.names

    def test_noop_action_name(self) -> None:
        env = self._make_env()
        assert env.noop_action_name == "NOOP"

    def test_env_id(self) -> None:
        env = self._make_env()
        assert env.env_id() == "glyphbench/atari-qbert-v0"

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
        dr = e1.action_spec.index_of("DOWN_RIGHT")
        actions = [noop, dr, dr, noop, dr]
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
        assert len(grid_lines) == 15
        lengths = [len(line) for line in grid_lines]
        assert len(set(lengths)) == 1

    def test_starts_at_top(self) -> None:
        env = self._make_env()
        env.reset(0)
        assert env._player_row == 0
        assert env._player_col == 0

    def test_cubes_created(self) -> None:
        env = self._make_env()
        env.reset(0)
        # 7-row pyramid: 1+2+3+4+5+6+7 = 28 cubes
        assert len(env._cubes) == 28
        assert all(not v for v in env._cubes.values())

    def test_hop_colors_cube(self) -> None:
        env = self._make_env()
        env.reset(0)
        dr = env.action_spec.index_of("DOWN_RIGHT")
        _, r, _, _, _ = env.step(dr)
        assert r >= 1  # +25 for coloring a cube

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
