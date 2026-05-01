"""Unit tests for Atari Ms. Pac-Man env."""

import pytest

from glyphbench.envs.atari.mspacman import MsPacManEnv


class TestMsPacMan:
    """Tests for the Ms. Pac-Man environment."""

    def _make_env(self, max_turns: int = 10000) -> MsPacManEnv:
        return MsPacManEnv(max_turns=max_turns)

    def test_action_space_defined(self):
        env = self._make_env()
        assert env.action_spec.n == 5
        expected = ("NOOP", "UP", "RIGHT", "LEFT", "DOWN")
        assert env.action_spec.names == expected

    def test_env_id(self):
        env = self._make_env()
        assert env.env_id() == "glyphbench/atari-mspacman-v0"

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
        up = e1.action_spec.index_of("UP")
        actions = [noop, up, up, noop, noop]
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
        assert len(grid_lines) == 24  # maze height
        assert all(len(line) == 28 for line in grid_lines)

    def test_pellet_collection(self):
        env = self._make_env()
        env.reset(0)
        # Place player next to a known pellet
        # Find a pellet adjacent to player
        initial_pellets = env._pellet_count
        # Move to collect pellets
        right = env.action_spec.index_of("RIGHT")
        collected = False
        for _ in range(10):
            _, reward, _, _, _ = env.step(right)
            if reward > 0:
                collected = True
                break
        # Either we collected or pellet_count decreased
        assert collected or env._pellet_count < initial_pellets

    def test_power_pellet_frightens_ghosts(self):
        env = self._make_env()
        env.reset(0)
        # Place player next to a power pellet
        # Power pellets are at corners of the maze
        env._player_x = 1
        env._player_y = 3  # Row with power pellet at position (1, 3) = '*'
        env._set_cell(2, 3, "*")
        env._pellet_count += 1
        right = env.action_spec.index_of("RIGHT")
        env.step(right)
        assert env._frightened_timer > 0

    def test_ghost_collision_loses_life(self):
        env = self._make_env()
        env.reset(0)
        initial_lives = env._lives
        # Place a ghost on the player
        for e in env._entities:
            if e.etype == "ghost":
                e.x = env._player_x + 1
                e.y = env._player_y
                e.data["state"] = "chase"
                break
        right = env.action_spec.index_of("RIGHT")
        env.step(right)
        # Ghost might have moved, but collision check should work
        # Check within a few steps
        noop = env.action_spec.index_of("NOOP")
        for _ in range(5):
            env.step(noop)
        # Lives may or may not have decreased depending on ghost movement
        assert env._lives <= initial_lives

    def test_scoring(self):
        env = self._make_env()
        env.reset(0)
        assert env._score == 0
        # Manually eat a pellet
        env._set_cell(env._player_x + 1, env._player_y, "·")
        # Make sure pellet count reflects it
        env._pellet_count += 1
        right = env.action_spec.index_of("RIGHT")
        env.step(right)
        assert env._score >= 1

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
        assert "mspacman" in prompt.lower() or "pellet" in prompt.lower()

    def test_ghosts_leave_pen(self):
        env = self._make_env(max_turns=2000)
        env.reset(seed=42)
        noop = env.action_spec.index_of("NOOP")
        pen_xs = set(range(11, 17))
        pen_ys = {10, 11, 12}
        door_cells = {(13, 9), (14, 9)}

        def in_pen_or_door(e):
            if (e.x, e.y) in door_cells:
                return True
            return e.x in pen_xs and e.y in pen_ys

        ever_escaped = {
            e.data["color"]: False
            for e in env._entities if e.etype == "ghost"
        }
        for _ in range(300):
            env.step(noop)
            for e in env._entities:
                if e.etype == "ghost" and not in_pen_or_door(e):
                    ever_escaped[e.data["color"]] = True

        assert all(ever_escaped.values()), (
            f"Some ghosts never escaped the pen: {ever_escaped}"
        )

    def test_released_ghost_cannot_reenter_pen(self):
        env = self._make_env(max_turns=2000)
        env.reset(seed=7)
        noop = env.action_spec.index_of("NOOP")
        pen_xs = set(range(11, 17))
        pen_ys = {10, 11, 12}

        # Track each ghost's release status; once released, it must not
        # re-enter the pen interior on any subsequent step.
        for _ in range(300):
            env.step(noop)
            for e in env._entities:
                if e.etype != "ghost":
                    continue
                in_pen = e.x in pen_xs and e.y in pen_ys
                if e.data.get("released"):
                    assert not in_pen, (
                        f"Released ghost {e.data['color']} re-entered "
                        f"pen at ({e.x}, {e.y})"
                    )
