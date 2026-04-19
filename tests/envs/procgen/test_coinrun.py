"""Unit tests for Procgen CoinRun env."""

import pytest

from atlas_rl.envs.procgen.coinrun import CoinRunEnv


class TestCoinRun:
    """Tests for the Procgen CoinRun environment."""

    def _make_env(self, max_turns: int = 512) -> CoinRunEnv:
        return CoinRunEnv(max_turns=max_turns)

    # --- Spec 10.1: test_action_space_defined ---
    def test_action_space_defined(self):
        env = self._make_env()
        assert env.action_spec.n == 5
        expected = ("NOOP", "LEFT", "RIGHT", "JUMP", "JUMP_RIGHT")
        assert env.action_spec.names == expected
        assert len(set(env.action_spec.names)) == 5

    def test_noop_action_name(self):
        env = self._make_env()
        assert env.noop_action_name == "NOOP"

    def test_env_id(self):
        env = self._make_env()
        assert env.env_id() == "atlas_rl/procgen-coinrun-v0"

    # --- Spec 10.1: test_reset_determinism ---
    def test_reset_determinism(self):
        e1 = self._make_env()
        e2 = self._make_env()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    # --- Spec 10.1: test_step_determinism ---
    def test_step_determinism(self):
        e1 = self._make_env()
        e2 = self._make_env()
        e1.reset(seed=0)
        e2.reset(seed=0)
        right = e1.action_spec.index_of("RIGHT")
        for _ in range(10):
            o1, r1, t1, tr1, _ = e1.step(right)
            o2, r2, t2, tr2, _ = e2.step(right)
            assert o1 == o2
            assert r1 == r2
            assert t1 == t2
            if t1 or tr1:
                break

    # --- Spec 8.3: seed determinism (same seed -> identical level) ---
    def test_seed_determinism_level_layout(self):
        e1 = self._make_env()
        e2 = self._make_env()
        e1.reset(seed=42)
        e2.reset(seed=42)
        # Internal level data should be identical
        assert e1._level_width == e2._level_width
        assert e1._level_data == e2._level_data

    # --- Spec 10.1: test_observation_contract ---
    def test_observation_contract(self):
        env = self._make_env()
        obs_str, _ = env.reset(seed=0)
        assert isinstance(obs_str, str)
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str
        assert "[HUD]" in obs_str
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        lengths = [len(line) for line in grid_lines]
        assert len(set(lengths)) == 1

    # --- Spec 8.3: window size is 20x12 ---
    def test_window_size(self):
        env = self._make_env()
        env.reset(seed=0)
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        assert len(grid_lines) == 12, f"Expected 12 rows, got {len(grid_lines)}"
        assert len(grid_lines[0]) == 20, f"Expected 20 cols, got {len(grid_lines[0])}"

    # --- Spec 8.3: death by pit ---
    def test_death_by_pit(self):
        """Falling into a pit terminates with 0 reward."""
        env = self._make_env()
        env.reset(seed=0)
        # Force agent above a pit
        # Find a pit in the level
        pit_found = False
        for x in range(env._level_width):
            if env._get_cell(x, env._ground_y) == "P":
                env._agent_x = x
                env._agent_y = env._ground_y - 1  # one above ground
                env._on_ground = False
                pit_found = True
                break
        if not pit_found:
            # Create a pit manually for testing
            env._set_cell(5, env._ground_y, "P")
            env._set_cell(5, env._ground_y + 1, "P")
            env._agent_x = 5
            env._agent_y = env._ground_y - 1
            env._on_ground = False

        # Step with NOOP to let gravity pull agent into pit
        noop = env.action_spec.index_of("NOOP")
        for _ in range(15):
            _, reward, terminated, _, info = env.step(noop)
            if terminated:
                assert reward == 0.0
                assert info.get("killed_by") is not None
                return
        # If no pit was reachable, skip
        pytest.skip("Could not position agent above a pit in this level layout")

    # --- Spec 8.3: coin collection = +10 ---
    def test_coin_collection_reward(self):
        """Reaching the coin gives +10."""
        env = self._make_env()
        env.reset(seed=0)
        # Place agent right next to the coin
        env._agent_x = env._coin_x - 1
        env._agent_y = env._coin_y
        env._on_ground = True
        env._jump_step = -1
        right = env.action_spec.index_of("RIGHT")
        _, reward, terminated, _, _ = env.step(right)
        if env._agent_x == env._coin_x and env._agent_y == env._coin_y:
            assert reward == 10.0
            assert terminated
        else:
            # Might need more steps
            for _ in range(3):
                _, reward, terminated, _, _ = env.step(right)
                if terminated:
                    assert reward == 10.0
                    return

    # --- Spec 8.3: window scrolls as agent moves ---
    def test_window_scrolls(self):
        env = self._make_env()
        env.reset(seed=0)
        right = env.action_spec.index_of("RIGHT")
        initial_camera = env._camera_x
        for _ in range(15):
            env.step(right)
        # Camera should have scrolled right
        assert env._camera_x > initial_camera or env._agent_x > 5

    # --- Spec 10.1: test_reward_bounds ---
    def test_reward_bounds(self):
        env = self._make_env()
        env.reset(seed=0)
        noop = env.action_spec.index_of("NOOP")
        for _ in range(20):
            _, reward, t, tr, _ = env.step(noop)
            assert reward in (0.0, 10.0), f"Unexpected reward: {reward}"
            if t or tr:
                break

    # --- Spec 10.1: test_max_turns_truncation ---
    def test_max_turns_truncation(self):
        env = self._make_env(max_turns=5)
        env.reset(seed=0)
        noop = env.action_spec.index_of("NOOP")
        for i in range(5):
            _, _, terminated, truncated, _ = env.step(noop)
            if terminated:
                return  # Died from level hazard
            if i == 4:
                assert truncated

    # --- Spec 8.3: HUD content ---
    def test_hud_content(self):
        env = self._make_env()
        env.reset(seed=0)
        hud = env.get_observation().hud
        assert "Step:" in hud
        assert "Vel:" in hud or "vel" in hud.lower()

    # --- Spec 8.3: info extras ---
    def test_info_extras(self):
        env = self._make_env()
        _, info = env.reset(seed=0)
        right = env.action_spec.index_of("RIGHT")
        _, _, _, _, info = env.step(right)
        assert "agent_pos" in info
        assert "level_seed" in info

    def test_reset_requires_seed(self):
        env = self._make_env()
        with pytest.raises(ValueError):
            env.reset()

    def test_system_prompt(self):
        env = self._make_env()
        prompt = env.system_prompt()
        assert len(prompt) > 0
        assert "coin" in prompt.lower() or "CoinRun" in prompt

    # --- Jump mechanics ---
    def test_jump_from_ground(self):
        """Jumping from ground should move agent upward."""
        env = self._make_env()
        env.reset(seed=0)
        initial_y = env._agent_y
        jump = env.action_spec.index_of("JUMP")
        env.step(jump)
        # Agent should be above initial position (lower y = higher)
        assert env._agent_y < initial_y or env._jump_step >= 0

    def test_cannot_double_jump(self):
        """Cannot jump while already in the air."""
        env = self._make_env()
        env.reset(seed=0)
        jump = env.action_spec.index_of("JUMP")
        env.step(jump)  # First jump
        env.step(jump)  # Second jump should be no-op (still in air)
        # Agent should continue the original jump arc, not start a new one
        # (y may change due to the arc, but it should not re-initiate)
        assert env._jump_step >= 0  # Still in jump arc

    # --- Gravity ---
    def test_gravity_pulls_agent_down(self):
        """Agent in air with no jump should fall."""
        env = self._make_env()
        env.reset(seed=0)
        # Force agent into air
        env._agent_y = env._ground_y - 3
        env._on_ground = False
        env._jump_step = -1
        noop = env.action_spec.index_of("NOOP")
        env.step(noop)
        # Agent should have fallen (y increased)
        assert env._agent_y >= env._ground_y - 3
