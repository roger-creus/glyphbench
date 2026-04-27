"""Unit tests for AtariBase via a minimal TestGame subclass."""

from __future__ import annotations

from typing import Any

import pytest

from glyphbench.core.action import ActionSpec
from glyphbench.envs.atari.base import AtariBase, AtariEntity


class TestGame(AtariBase):
    """Minimal concrete subclass used only for testing AtariBase."""

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "shoot",
        ),
    )

    def env_id(self) -> str:
        return "glyphbench/test-game-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(10, 8)
        self._player_x = 5
        self._player_y = 6

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        if action_name == "LEFT":
            self._player_x = max(0, self._player_x - 1)
        elif action_name == "RIGHT":
            self._player_x = min(self._grid_w - 1, self._player_x + 1)
        elif action_name == "FIRE":
            self._add_entity("bullet", "*", self._player_x, self._player_y - 1, dy=-1)
        return 0.0, False, {}

    def _render_current_observation(self):  # type: ignore[override]
        # Delegate to AtariBase default renderer.
        return super()._render_current_observation()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_env(max_turns: int = 1000) -> TestGame:
    return TestGame(max_turns=max_turns)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAtariBaseInit:
    def test_initial_state_before_reset(self) -> None:
        env = make_env()
        assert env._score == 0
        # GlyphBench uses single-life Atari (death = episode terminates).
        assert env._lives == 1
        assert env._level == 1
        assert env._game_over is False
        assert env._entities == []
        assert env._message == ""

    def test_action_spec_accessible(self) -> None:
        env = make_env()
        assert env.action_spec.n == 4
        assert "NOOP" in env.action_spec.names

    def test_noop_action_name(self) -> None:
        env = make_env()
        assert env.noop_action_name == "NOOP"

    def test_env_id(self) -> None:
        env = make_env()
        assert env.env_id() == "glyphbench/test-game-v0"


class TestAtariBaseGrid:
    def test_init_grid_dimensions(self) -> None:
        env = make_env()
        env.reset(0)
        assert env._grid_w == 10
        assert env._grid_h == 8
        assert len(env._grid) == 8
        assert all(len(row) == 10 for row in env._grid)

    def test_init_grid_fill(self) -> None:
        env = make_env()
        env.reset(0)
        # Default fill is space
        assert all(env._grid[y][x] == " " for y in range(8) for x in range(10))

    def test_set_cell_and_grid_at(self) -> None:
        env = make_env()
        env.reset(0)
        env._set_cell(3, 4, "\u2588")
        assert env._grid_at(3, 4) == "\u2588"

    def test_set_cell_out_of_bounds_noop(self) -> None:
        env = make_env()
        env.reset(0)
        # Should not raise
        env._set_cell(99, 99, "X")
        env._set_cell(-1, -1, "X")

    def test_grid_at_out_of_bounds_returns_hash(self) -> None:
        env = make_env()
        env.reset(0)
        assert env._grid_at(-1, 0) == "\u2588"
        assert env._grid_at(0, -1) == "\u2588"
        assert env._grid_at(100, 0) == "\u2588"

    def test_is_solid_wall_chars(self) -> None:
        env = make_env()
        env.reset(0)
        env._set_cell(0, 0, "█")
        assert env._is_solid(0, 0) is True
        env._set_cell(1, 0, " ")
        assert env._is_solid(1, 0) is False


class TestAtariBaseReset:
    def test_reset_returns_string(self) -> None:
        env = make_env()
        obs, info = env.reset(7)
        assert isinstance(obs, str)

    def test_reset_clears_score_lives_entities(self) -> None:
        env = make_env()
        env.reset(0)
        # Manually dirty the state
        env._score = 100
        env._lives = 0
        env._entities.append(
            AtariEntity(etype="enemy", char="E", x=2, y=2)
        )
        env.reset(0)
        assert env._score == 0
        # Single-life model: reset re-arms exactly one life.
        assert env._lives == 1
        assert env._entities == []

    def test_reset_determinism(self) -> None:
        e1 = make_env()
        e2 = make_env()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    def test_reset_different_seeds_differ(self) -> None:
        # TestGame ignores seed, so same obs; just verify no crash with different seeds
        env = make_env()
        env.reset(1)
        env.reset(2)

    def test_reset_requires_seed(self) -> None:
        env = make_env()
        with pytest.raises(TypeError):
            env.reset()  # type: ignore[call-arg]


class TestAtariBaseObservation:
    def test_observation_has_required_sections(self) -> None:
        env = make_env()
        obs, _ = env.reset(0)
        assert "[Grid]" in obs
        assert "[Legend]" in obs
        assert "[HUD]" in obs

    def test_hud_format(self) -> None:
        env = make_env()
        env.reset(0)
        obs_obj = env.get_observation()
        hud = obs_obj.hud
        # Single-life model: HUD has Score + Level (no Lives field).
        # HUD is computed for info-dict only — never shown to the model.
        assert "Score:" in hud
        assert "Level:" in hud
        assert "Lives" not in hud

    def test_hud_reflects_score_and_level(self) -> None:
        env = make_env()
        env.reset(0)
        env._score = 42
        env._level = 3
        obs_obj = env.get_observation()
        assert "42" in obs_obj.hud
        assert "3" in obs_obj.hud

    def test_player_shown_as_at_sign(self) -> None:
        env = make_env()
        env.reset(0)
        grid_str = env.get_observation().grid
        assert "@" in grid_str

    def test_legend_contains_player_symbol(self) -> None:
        env = make_env()
        env.reset(0)
        legend = env.get_observation().legend
        assert "@" in legend

    def test_grid_rows_equal_length(self) -> None:
        env = make_env()
        env.reset(0)
        grid_str = env.get_observation().grid
        rows = grid_str.split("\n")
        lengths = [len(r) for r in rows]
        assert len(set(lengths)) == 1, f"Variable row lengths: {lengths}"


class TestAtariBaseScoreAndLives:
    def test_on_point_scored_increases_score(self) -> None:
        env = make_env()
        env.reset(0)
        env._on_point_scored(5)
        assert env._score == 5
        env._on_point_scored(3)
        assert env._score == 8

    def test_on_life_lost_terminates_immediately(self) -> None:
        # Single-life model: any life-loss event ends the episode.
        env = make_env()
        env.reset(0)
        env._on_life_lost()
        assert env._lives == 0
        assert env._game_over is True
        assert env._message == "Game Over!"

    def test_game_over_terminates_step(self) -> None:
        env = make_env()
        env.reset(0)
        env._on_life_lost()
        noop = env.action_spec.index_of("NOOP")
        _, _, terminated, _, _ = env.step(noop)
        assert terminated is True


class TestAtariBaseEntities:
    def test_add_entity_appends(self) -> None:
        env = make_env()
        env.reset(0)
        e = env._add_entity("enemy", "E", 3, 4)
        assert len(env._entities) == 1
        assert e.etype == "enemy"
        assert e.char == "E"
        assert e.x == 3
        assert e.y == 4

    def test_add_entity_with_velocity(self) -> None:
        env = make_env()
        env.reset(0)
        e = env._add_entity("bullet", "*", 5, 5, dx=1, dy=-1)
        assert e.dx == 1
        assert e.dy == -1

    def test_advance_entities_moves_them(self) -> None:
        env = make_env()
        env.reset(0)
        env._add_entity("bullet", "*", 3, 3, dx=1, dy=0)
        env._advance_entities()
        assert env._entities[0].x == 4
        assert env._entities[0].y == 3

    def test_advance_entities_removes_out_of_bounds(self) -> None:
        env = make_env()
        env.reset(0)
        # Place entity at edge heading out
        env._add_entity("bullet", "*", 9, 3, dx=1, dy=0)
        env._advance_entities()
        assert len(env._entities) == 0

    def test_advance_entities_skips_dead(self) -> None:
        env = make_env()
        env.reset(0)
        e = env._add_entity("enemy", "E", 3, 3, dx=0, dy=0)
        e.alive = False
        env._advance_entities()
        assert len(env._entities) == 0

    def test_entities_rendered_in_grid(self) -> None:
        env = make_env()
        env.reset(0)
        env._add_entity("enemy", "E", 2, 2)
        grid_str = env.get_observation().grid
        assert "E" in grid_str

    def test_fire_action_adds_bullet_entity(self) -> None:
        env = make_env()
        env.reset(0)
        fire = env.action_spec.index_of("FIRE")
        env.step(fire)
        assert any(e.char == "*" for e in env._entities)


class TestAtariBaseStep:
    def test_step_info_contains_score_and_level(self) -> None:
        # Single-life model: info no longer carries a "lives" key.
        env = make_env()
        env.reset(0)
        noop = env.action_spec.index_of("NOOP")
        _, _, _, _, info = env.step(noop)
        assert "score" in info
        assert "level" in info
        assert "lives" not in info

    def test_step_advances_entities(self) -> None:
        env = make_env()
        env.reset(0)
        env._add_entity("bullet", "*", 3, 3, dx=1, dy=0)
        noop = env.action_spec.index_of("NOOP")
        env.step(noop)
        remaining = [e for e in env._entities if e.char == "*"]
        if remaining:
            assert remaining[0].x == 4

    def test_step_message_cleared_each_turn(self) -> None:
        env = make_env()
        env.reset(0)
        env._message = "old message"
        noop = env.action_spec.index_of("NOOP")
        env.step(noop)
        # After step, _message should have been reset to "" at start of _step
        # (it may be set again by _game_step, but TestGame doesn't set it)
        assert env._message == ""

    def test_max_turns_truncation(self) -> None:
        env = make_env(max_turns=3)
        env.reset(0)
        noop = env.action_spec.index_of("NOOP")
        for i in range(3):
            _, _, terminated, truncated, _ = env.step(noop)
            if terminated:
                return
            if i == 2:
                assert truncated

    def test_left_right_moves_player(self) -> None:
        env = make_env()
        env.reset(0)
        initial_x = env._player_x
        right = env.action_spec.index_of("RIGHT")
        env.step(right)
        assert env._player_x == initial_x + 1

        left = env.action_spec.index_of("LEFT")
        env.step(left)
        assert env._player_x == initial_x


class TestAtariBaseSystemPrompt:
    def test_system_prompt_contains_env_id(self) -> None:
        env = make_env()
        env.reset(0)
        prompt = env.system_prompt()
        assert "test-game-v0" in prompt

    def test_system_prompt_contains_actions(self) -> None:
        env = make_env()
        env.reset(0)
        prompt = env.system_prompt()
        assert "NOOP" in prompt
        assert "LEFT" in prompt
        assert "RIGHT" in prompt
        assert "FIRE" in prompt
