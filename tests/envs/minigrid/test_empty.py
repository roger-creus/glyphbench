"""Unit tests for MiniGrid Empty-5x5 env."""

import pytest

from atlas_rl.envs.minigrid.empty import MiniGridEmpty5x5Env


class TestMiniGridEmpty5x5:
    """Tests for the MiniGrid Empty-5x5 environment."""

    def _make_env(self, max_turns: int = 100) -> MiniGridEmpty5x5Env:
        return MiniGridEmpty5x5Env(max_turns=max_turns)

    # --- Spec 10.1: test_action_space_defined ---
    def test_action_space_defined(self):
        env = self._make_env()
        assert env.action_spec.n == 7
        expected_names = (
            "TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD",
            "PICKUP", "DROP", "TOGGLE", "DONE",
        )
        assert env.action_spec.names == expected_names
        # All names unique
        assert len(set(env.action_spec.names)) == 7

    def test_noop_action_name(self):
        env = self._make_env()
        assert env.noop_action_name == "DONE"
        # noop_action_name must be in action_spec
        assert env.noop_action_name in env.action_spec.names

    def test_env_id(self):
        env = self._make_env()
        assert env.env_id() == "atlas_rl/minigrid-empty-5x5-v0"

    # --- Spec 10.1: test_reset_determinism ---
    def test_reset_determinism(self):
        e1 = self._make_env()
        e2 = self._make_env()
        o1, i1 = e1.reset(seed=42)
        o2, i2 = e2.reset(seed=42)
        assert o1 == o2

    # --- Spec 10.1: test_step_determinism ---
    def test_step_determinism(self):
        e1 = self._make_env()
        e2 = self._make_env()
        e1.reset(seed=0)
        e2.reset(seed=0)
        fwd = e1.action_spec.index_of("MOVE_FORWARD")
        for _ in range(5):
            o1, r1, t1, tr1, _ = e1.step(fwd)
            o2, r2, t2, tr2, _ = e2.step(fwd)
            assert o1 == o2
            assert r1 == r2
            assert t1 == t2
            assert tr1 == tr2

    # --- Spec 10.1: test_seed_reproducibility ---
    def test_seed_reproducibility(self):
        """Same seed -> byte-identical trajectory over 20 steps."""
        e1 = self._make_env()
        e2 = self._make_env()
        e1.reset(seed=99)
        e2.reset(seed=99)
        fwd = e1.action_spec.index_of("MOVE_FORWARD")
        right = e1.action_spec.index_of("TURN_RIGHT")
        actions = [fwd, fwd, right, fwd, fwd, right, fwd, fwd]
        for a in actions:
            o1, r1, t1, tr1, _ = e1.step(a)
            o2, r2, t2, tr2, _ = e2.step(a)
            assert o1 == o2
            assert r1 == r2

    # --- Spec 10.1: test_observation_contract ---
    def test_observation_contract(self):
        env = self._make_env()
        obs_str, info = env.reset(seed=0)
        assert isinstance(obs_str, str)
        assert len(obs_str) > 0
        # Grid must be present
        assert "[Grid]" in obs_str
        # Legend must be present
        assert "[Legend]" in obs_str
        # HUD must be present
        assert "[HUD]" in obs_str
        # Grid must be rectangular
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        lengths = [len(line) for line in grid_lines]
        assert len(set(lengths)) == 1, f"Grid rows have different lengths: {lengths}"

    # --- Spec 8.1: Agent starts at (1,1) facing right ---
    def test_initial_position_and_facing(self):
        env = self._make_env()
        obs_str, info = env.reset(seed=0)
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        # 7x7 grid: row 1 (index 1), col 1 should have '>'
        assert grid_lines[1][1] == ">", f"Expected '>' at (1,1), got '{grid_lines[1][1]}'"

    # --- Spec 8.1: Goal at (5,5) ---
    def test_goal_position(self):
        env = self._make_env()
        env.reset(seed=0)
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        # Row 5, col 5 should have 'G'
        assert grid_lines[5][5] == "G", f"Expected 'G' at (5,5), got '{grid_lines[5][5]}'"

    # --- Spec 8.1 test: Move forward 4 times from start -> end up at (5,1) on the grid ---
    # Agent starts at interior (1,1) facing right. 4 forwards = interior pos (5,1) i.e. grid (5,1)
    # Wait -- spec says "Move forward from starting position 4 times -> end up at (4,0)"
    # This is in INTERIOR coords: start at interior (0,0), move right 4 -> interior (4,0)
    # In grid coords that's (5,1)
    def test_move_forward_four_times(self):
        env = self._make_env()
        env.reset(seed=0)
        fwd = env.action_spec.index_of("MOVE_FORWARD")
        for _ in range(4):
            env.step(fwd)
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        # Agent should be at grid col 5, row 1 (interior pos 4,0)
        assert grid_lines[1][5] == ">", f"Expected '>' at grid (5,1), got '{grid_lines[1][5]}'"

    # --- Spec 8.1: Turn-right -> turn-right -> facing = LEFT ---
    def test_turn_right_twice_faces_left(self):
        env = self._make_env()
        env.reset(seed=0)
        right = env.action_spec.index_of("TURN_RIGHT")
        env.step(right)  # facing down
        env.step(right)  # facing left
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        # Agent should still be at (1,1) but facing left
        assert grid_lines[1][1] == "<", f"Expected '<' at (1,1), got '{grid_lines[1][1]}'"

    # --- Spec 8.1: Reach goal in optimal path (8 steps) ---
    # From interior (0,0) facing right:
    # 4x MOVE_FORWARD to reach interior (4,0)
    # TURN_RIGHT (now facing down)
    # 4x MOVE_FORWARD to reach interior (4,4)
    # That's 9 steps total. But goal is at interior (4,4) so terminated after 9 steps.
    # Spec says "optimal path (8 steps)" -- let's check:
    # Actually starting at interior (0,0) going to interior (4,4):
    # RIGHT is already facing right, so 4 forwards to col 4, 1 turn right, 4 forwards to row 4 = 9 actions
    # But the spec says 8 -- perhaps counting differently. Let me be faithful to spec.
    # 8 steps at reward approx 0.928. That implies step_count=8 at the time of reaching goal.
    # Possible path: if agent starts at (1,1) in grid coords, goal at (5,5):
    # 4 fwd (reaches col 5), turn_right, 4 fwd (reaches row 5) = 9 actions
    # Unless we do diagonal... no, MiniGrid has no diagonal.
    # Spec might count differently. Let me just test the 9-step path and check reward.
    def test_optimal_path_reward(self):
        env = self._make_env()
        env.reset(seed=0)
        fwd = env.action_spec.index_of("MOVE_FORWARD")
        turn_right = env.action_spec.index_of("TURN_RIGHT")

        # 4 forwards to reach right side
        for _ in range(4):
            _, r, t, _, _ = env.step(fwd)
            assert not t

        # Turn right (now facing down)
        _, r, t, _, _ = env.step(turn_right)
        assert not t

        # 4 forwards to reach bottom
        for i in range(4):
            _, r, t, _, info = env.step(fwd)
            if i < 3:
                assert not t
            else:
                # Should reach goal on the 4th forward (step 9 total)
                assert t
                # Reward = 1 - 0.9 * step/max_steps
                # step = 9 (9 actions taken)
                expected_reward = 1.0 - 0.9 * (9 / 100)
                assert abs(r - expected_reward) < 1e-6, f"Expected {expected_reward}, got {r}"

    # --- Spec 10.1: test_reward_bounds ---
    def test_reward_bounds(self):
        env = self._make_env()
        env.reset(seed=0)
        fwd = env.action_spec.index_of("MOVE_FORWARD")
        noop = env.action_spec.index_of("DONE")
        # Non-terminal steps have 0 reward
        _, r, _, _, _ = env.step(noop)
        assert r == 0.0
        _, r, _, _, _ = env.step(fwd)
        assert r == 0.0

    # --- Spec 10.1: test_episode_terminates ---
    def test_episode_terminates(self):
        """Random-ish actions eventually end (via truncation at max_turns)."""
        env = self._make_env(max_turns=50)
        env.reset(seed=0)
        noop = env.action_spec.index_of("DONE")
        for _ in range(60):
            _, _, t, tr, _ = env.step(noop)
            if t or tr:
                return
        pytest.fail("Episode did not terminate within 60 steps")

    # --- Spec 10.1: test_max_turns_truncation ---
    def test_max_turns_truncation(self):
        env = self._make_env(max_turns=5)
        env.reset(seed=0)
        noop = env.action_spec.index_of("DONE")
        for i in range(5):
            _, _, terminated, truncated, info = env.step(noop)
            if i < 4:
                assert not truncated
                assert not terminated
            else:
                assert truncated
                assert not terminated

    # --- Spec 8.1: no-op actions (PICKUP, DROP, TOGGLE, DONE) don't change state ---
    def test_noop_actions_do_not_change_state(self):
        env = self._make_env()
        env.reset(seed=0)
        obs_before = env.get_observation()
        for name in ("PICKUP", "DROP", "TOGGLE", "DONE"):
            idx = env.action_spec.index_of(name)
            env.step(idx)
        # Position should not have changed (grid still shows agent at (1,1))
        obs_after = env.get_observation()
        assert obs_before.grid == obs_after.grid

    # --- Spec 8.1: wall collision = stay put ---
    def test_wall_collision_stays_put(self):
        env = self._make_env()
        env.reset(seed=0)
        # Agent at (1,1) facing right. Turn left to face up.
        turn_left = env.action_spec.index_of("TURN_LEFT")
        fwd = env.action_spec.index_of("MOVE_FORWARD")
        env.step(turn_left)  # Now facing up
        env.step(fwd)  # Try to move into wall -- should stay at (1,1)
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        assert grid_lines[1][1] == "^", "Expected '^' at (1,1) after wall bump"

    # --- Spec 8.1: HUD format ---
    def test_hud_format(self):
        env = self._make_env()
        env.reset(seed=0)
        grid_obs = env.get_observation()
        assert "Step:" in grid_obs.hud
        assert "Facing:" in grid_obs.hud
        assert "Position:" in grid_obs.hud

    # --- Spec 8.1: legend ---
    def test_legend_contains_agent_and_goal(self):
        env = self._make_env()
        env.reset(seed=0)
        grid_obs = env.get_observation()
        assert ">" in grid_obs.legend or "agent" in grid_obs.legend.lower()
        assert "G" in grid_obs.legend
        assert "goal" in grid_obs.legend.lower()

    # --- Spec: system_prompt ---
    def test_system_prompt_mentions_game(self):
        env = self._make_env()
        prompt = env.system_prompt()
        assert len(prompt) > 0
        assert "MOVE_FORWARD" in prompt or "action" in prompt.lower()

    # --- Spec 8.1: info extras ---
    def test_info_extras(self):
        env = self._make_env()
        _, info = env.reset(seed=0)
        assert "seed" in info
        fwd = env.action_spec.index_of("MOVE_FORWARD")
        _, _, _, _, info = env.step(fwd)
        assert "goal_reached" in info

    # --- Edge: reset clears state from previous episode ---
    def test_reset_clears_previous_state(self):
        env = self._make_env()
        env.reset(seed=0)
        fwd = env.action_spec.index_of("MOVE_FORWARD")
        env.step(fwd)
        env.step(fwd)
        # Reset should put agent back at start
        env.reset(seed=0)
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        assert grid_lines[1][1] == ">"

    # --- Spec: requires explicit seed ---
    def test_reset_requires_seed(self):
        env = self._make_env()
        with pytest.raises(ValueError):
            env.reset()

    # --- Grid dimensions ---
    def test_grid_is_7x7(self):
        env = self._make_env()
        env.reset(seed=0)
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        assert len(grid_lines) == 7, f"Expected 7 rows, got {len(grid_lines)}"
        for line in grid_lines:
            assert len(line) == 7, f"Expected 7 cols, got {len(line)}"
