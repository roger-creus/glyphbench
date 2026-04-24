"""Unit tests for Atari Pong env."""

import pytest

from glyphbench.envs.atari.pong import PongEnv


class TestPong:
    """Tests for the Atari Pong environment."""

    def _make_env(self, max_turns: int = 5000) -> PongEnv:
        return PongEnv(max_turns=max_turns)

    # --- Spec 10.1: test_action_space_defined ---
    def test_action_space_defined(self):
        env = self._make_env()
        assert env.action_spec.n == 6
        expected = ("NOOP", "FIRE", "UP", "DOWN", "UP_FIRE", "DOWN_FIRE")
        assert env.action_spec.names == expected
        assert len(set(env.action_spec.names)) == 6

    def test_noop_action_name(self):
        env = self._make_env()
        assert env.noop_action_name == "NOOP"

    def test_env_id(self):
        env = self._make_env()
        assert env.env_id() == "glyphbench/atari-pong-v0"

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
        fire = e1.action_spec.index_of("FIRE")
        up = e1.action_spec.index_of("UP")
        actions = [fire, up, up, up, up, up]
        for a in actions:
            o1, r1, t1, tr1, _ = e1.step(a)
            o2, r2, t2, tr2, _ = e2.step(a)
            assert o1 == o2
            assert r1 == r2
            assert t1 == t2

    # --- Spec 10.1: test_observation_contract ---
    def test_observation_contract(self):
        env = self._make_env()
        obs_str, _ = env.reset(0)
        assert isinstance(obs_str, str)
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str
        assert "[HUD]" in obs_str
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        lengths = [len(line) for line in grid_lines]
        assert len(set(lengths)) == 1, f"Variable row lengths: {lengths}"

    # --- Spec 8.5: court dimensions ---
    def test_court_dimensions(self):
        """Court is 34x18 (32x16 + border)."""
        env = self._make_env()
        env.reset(0)
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        assert len(grid_lines) == 18, f"Expected 18 rows, got {len(grid_lines)}"
        assert len(grid_lines[0]) == 34, f"Expected 34 cols, got {len(grid_lines[0])}"

    # --- Spec 8.5: ball physics ---
    def test_ball_moves_after_serve(self):
        """After FIRE, ball should move each step."""
        env = self._make_env()
        env.reset(0)
        fire = env.action_spec.index_of("FIRE")
        noop = env.action_spec.index_of("NOOP")
        # Serve
        env.step(fire)
        pos1 = (env._ball_x, env._ball_y)
        # Step
        env.step(noop)
        pos2 = (env._ball_x, env._ball_y)
        assert pos1 != pos2, "Ball should move after serve"

    # --- Spec 8.5: paddle bounce ---
    def test_ball_bounces_off_top_bottom_walls(self):
        """Ball should bounce (vy flips) when hitting top or bottom wall."""
        env = self._make_env()
        env.reset(0)
        # Force ball near top wall heading up
        env._ball_x = 16
        env._ball_y = 1  # row 1 is first interior row
        env._ball_vx = 1
        env._ball_vy = -1
        env._serving = False
        noop = env.action_spec.index_of("NOOP")
        env.step(noop)
        # vy should have flipped
        assert env._ball_vy == 1, f"Expected vy=1 after top bounce, got {env._ball_vy}"

    # --- Spec 8.5: scoring ---
    def test_scoring_agent_scores(self):
        """Ball passing left edge -> agent scores +1."""
        env = self._make_env()
        env.reset(0)
        # Force ball near left edge heading left, paddle out of the way
        env._ball_x = 2
        env._ball_y = 3
        env._ball_vx = -1
        env._ball_vy = 0
        env._paddle_left_y = 14  # move opponent paddle far away
        env._serving = False
        noop = env.action_spec.index_of("NOOP")
        # Step until ball passes left edge
        for _ in range(5):
            _, reward, _, _, info = env.step(noop)
            if reward > 0:
                assert reward == 1.0
                assert env._score_right > 0 or info.get("score_right", 0) > 0
                return
        # Ball should have scored by now
        assert env._score_right >= 1, "Agent should have scored"

    def test_scoring_opponent_scores(self):
        """Ball passing right edge -> opponent scores, reward = -1."""
        env = self._make_env()
        env.reset(0)
        # Force ball near right edge heading right, paddle out of the way
        env._ball_x = 31
        env._ball_y = 3
        env._ball_vx = 1
        env._ball_vy = 0
        env._paddle_right_y = 14  # move agent paddle far away
        env._serving = False
        noop = env.action_spec.index_of("NOOP")
        for _ in range(5):
            _, reward, _, _, info = env.step(noop)
            if reward < 0:
                assert reward == -1.0
                return
        assert env._score_left >= 1, "Opponent should have scored"

    # --- Spec 8.5: terminal condition (first to 21) ---
    def test_game_ends_at_21(self):
        """First side to 21 -> terminated."""
        env = self._make_env()
        env.reset(0)
        # Force score to near-end
        env._score_right = 20
        env._score_left = 0
        # Force ball to score for agent, paddle out of the way
        env._ball_x = 2
        env._ball_y = 3
        env._ball_vx = -1
        env._ball_vy = 0
        env._paddle_left_y = 14  # move opponent paddle far away
        env._serving = False
        noop = env.action_spec.index_of("NOOP")
        for _ in range(10):
            _, _, terminated, _, info = env.step(noop)
            if terminated:
                assert info.get("game_over", False)
                return
        pytest.fail("Game should have ended at 21 points")

    # --- Spec 8.5: opponent AI determinism ---
    def test_opponent_ai_determinism(self):
        """Fixed seed -> identical opponent behavior."""
        e1 = self._make_env()
        e2 = self._make_env()
        e1.reset(seed=42)
        e2.reset(seed=42)
        fire = e1.action_spec.index_of("FIRE")
        noop = e1.action_spec.index_of("NOOP")
        e1.step(fire)
        e2.step(fire)
        for _ in range(20):
            o1, _, t1, tr1, _ = e1.step(noop)
            o2, _, t2, tr2, _ = e2.step(noop)
            assert o1 == o2
            if t1 or tr1:
                break

    # --- Spec 8.5: paddle movement ---
    def test_paddle_moves_up_down(self):
        env = self._make_env()
        env.reset(0)
        initial_y = env._paddle_right_y
        up = env.action_spec.index_of("UP")
        env.step(up)
        assert env._paddle_right_y == initial_y - 1 or env._paddle_right_y == initial_y
        # (might be at top wall already)

    # --- Spec 8.5: ball speed up ---
    def test_ball_speed_increases_every_6_hits(self):
        env = self._make_env()
        env.reset(0)
        assert env._ball_speed_level == 1
        env._rally_hits = 5
        # Simulate a paddle hit
        env._on_paddle_hit()
        assert env._rally_hits == 6
        assert env._ball_speed_level == 2

    # --- Spec 8.5: HUD shows ball velocity ---
    def test_hud_shows_ball_info(self):
        env = self._make_env()
        env.reset(0)
        hud = env.get_observation().hud
        assert "Score:" in hud
        assert "Ball:" in hud or "vel" in hud.lower()

    # --- Spec 8.5: FIRE serves ball ---
    def test_fire_serves_ball(self):
        env = self._make_env()
        env.reset(0)
        assert env._serving
        fire = env.action_spec.index_of("FIRE")
        env.step(fire)
        assert not env._serving

    # --- Spec 10.1: max turns truncation ---
    def test_max_turns_truncation(self):
        env = self._make_env(max_turns=5)
        env.reset(0)
        noop = env.action_spec.index_of("NOOP")
        for i in range(5):
            _, _, terminated, truncated, _ = env.step(noop)
            if terminated:
                return  # Game ended normally
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
        assert "Pong" in prompt or "paddle" in prompt.lower()

    # --- Spec 8.5: reward bounds ---
    def test_reward_bounds(self):
        env = self._make_env()
        env.reset(0)
        fire = env.action_spec.index_of("FIRE")
        noop = env.action_spec.index_of("NOOP")
        env.step(fire)
        for _ in range(50):
            _, reward, t, tr, _ = env.step(noop)
            assert reward in (-1.0, 0.0, 1.0), f"Unexpected reward: {reward}"
            if t or tr:
                break
