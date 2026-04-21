"""Unit tests for Atari Surround env."""

from glyphbench.envs.atari.surround import SurroundEnv


class TestSurround:
    """Tests for the Surround environment."""

    def _make_env(self, max_turns: int = 10000) -> SurroundEnv:
        return SurroundEnv(max_turns=max_turns)

    def test_action_space_defined(self) -> None:
        env = self._make_env()
        assert env.action_spec.n == 5
        assert "NOOP" in env.action_spec.names
        assert "UP" in env.action_spec.names
        assert "RIGHT" in env.action_spec.names
        assert "LEFT" in env.action_spec.names
        assert "DOWN" in env.action_spec.names

    def test_noop_action_name(self) -> None:
        env = self._make_env()
        assert env.noop_action_name == "NOOP"

    def test_env_id(self) -> None:
        env = self._make_env()
        assert env.env_id() == "glyphbench/atari-surround-v0"

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
        actions = [noop, up, noop, noop, up]
        for a in actions:
            o1, r1, t1, tr1, _ = e1.step(a)
            o2, r2, t2, tr2, _ = e2.step(a)
            assert o1 == o2
            assert r1 == r2
            assert t1 == t2
            if t1:
                break

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

    def test_players_start_on_opposite_sides(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        assert env._player_x < env._opp_x

    def test_trails_start_with_one_cell(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        assert len(env._trail_player) == 1
        assert len(env._trail_opp) == 1

    def test_no_180_turn(self) -> None:
        """Player cannot reverse direction."""
        env = self._make_env()
        env.reset(seed=0)
        # Player starts moving right; LEFT should not reverse
        assert env._player_dx == 1
        left = env.action_spec.index_of("LEFT")
        env.step(left)
        # Direction should still be right (no 180)
        assert env._player_dx == 1

    def test_terminates_on_crash(self) -> None:
        """Game should terminate when someone crashes."""
        env = self._make_env(max_turns=500)
        env.reset(seed=42)
        noop = env.action_spec.index_of("NOOP")
        terminated = False
        for _ in range(500):
            _, _, terminated, truncated, _ = env.step(noop)
            if terminated or truncated:
                break
        assert terminated  # should crash into wall eventually

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
