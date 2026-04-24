"""Tests for MiniHack Boxoban environments."""

from __future__ import annotations

import pytest

from glyphbench.envs.minihack.boxoban import (
    MiniHackBoxobanHardEnv,
    MiniHackBoxobanMediumEnv,
    MiniHackBoxobanUnfilteredEnv,
)

BOXOBAN_CLASSES = [
    MiniHackBoxobanMediumEnv,
    MiniHackBoxobanHardEnv,
    MiniHackBoxobanUnfilteredEnv,
]
BOXOBAN_IDS = [
    "glyphbench/minihack-boxoban-medium-v0",
    "glyphbench/minihack-boxoban-hard-v0",
    "glyphbench/minihack-boxoban-unfiltered-v0",
]


class TestBoxobanEnvs:
    """Tests for all Boxoban variants."""

    @pytest.mark.parametrize(
        "cls,expected_id", zip(BOXOBAN_CLASSES, BOXOBAN_IDS, strict=True)
    )
    def test_env_id(self, cls: type, expected_id: str) -> None:
        env = cls()
        assert env.env_id() == expected_id

    @pytest.mark.parametrize("cls", BOXOBAN_CLASSES)
    def test_reset_determinism(self, cls: type) -> None:
        e1, e2 = cls(), cls()
        o1, _ = e1.reset(42)
        o2, _ = e2.reset(42)
        assert o1 == o2

    @pytest.mark.parametrize("cls", BOXOBAN_CLASSES)
    def test_reset_produces_valid_grid(self, cls: type) -> None:
        env = cls()
        obs_str, _ = env.reset(0)
        assert "@" in obs_str
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str
        assert "[HUD]" in obs_str

    @pytest.mark.parametrize("cls", BOXOBAN_CLASSES)
    def test_grid_rows_same_length(self, cls: type) -> None:
        env = cls()
        env.reset(0)
        grid_obs = env.get_observation()
        lines = grid_obs.grid.split("\n")
        lengths = [len(line) for line in lines]
        assert len(set(lengths)) == 1, f"Unequal row lengths: {lengths}"

    @pytest.mark.parametrize("cls", BOXOBAN_CLASSES)
    def test_boxes_and_targets_present(self, cls: type) -> None:
        env = cls()
        obs_str, _ = env.reset(0)
        # Must have boxes (0) and/or targets (X) and/or box-on-target (*)
        has_box = "0" in obs_str or "*" in obs_str
        has_target = "X" in obs_str or "*" in obs_str
        assert has_box, "No boxes found in observation"
        assert has_target, "No targets found in observation"

    @pytest.mark.parametrize("cls", BOXOBAN_CLASSES)
    def test_box_count_matches(self, cls: type) -> None:
        env = cls()
        env.reset(0)
        assert len(env._box_positions) == env._num_boxes
        assert len(env._target_positions) == env._num_boxes

    @pytest.mark.parametrize("cls", BOXOBAN_CLASSES)
    def test_push_box(self, cls: type) -> None:
        """Place player adjacent to a box and push it."""
        env = cls(max_turns=100)
        env.reset(0)

        # Manually set up a pushable scenario:
        # Player at (3,3), box at (4,3), empty at (5,3)
        s = env._grid_size
        env._player_pos = (2, s // 2)
        env._box_positions = [(3, s // 2)]
        env._target_positions = [(4, s // 2)]
        # Ensure cells are walkable
        env._grid[s // 2][3] = "·"
        env._grid[s // 2][4] = "·"

        move_e = env.action_spec.index_of("MOVE_E")
        _, _, terminated, _, info = env.step(move_e)

        # Box should have moved east
        assert env._box_positions[0] == (4, s // 2)
        assert env._player_pos == (3, s // 2)
        # Box is now on target -> win
        assert terminated
        assert info["boxes_on_target"] == 1

    @pytest.mark.parametrize("cls", BOXOBAN_CLASSES)
    def test_cannot_push_box_into_wall(self, cls: type) -> None:
        """A box against a wall should not be pushable further."""
        env = cls(max_turns=100)
        env.reset(0)
        s = env._grid_size

        # Player at (s-4, 1), box at (s-3, 1) -- wall at (s-2, 1) actually
        # Use border wall at x = s-1
        env._player_pos = (s - 3, 1)
        env._box_positions = [(s - 2, 1)]  # next to right border wall
        env._target_positions = [(1, 1)]
        env._grid[1][s - 2] = "·"

        old_box = env._box_positions[0]
        move_e = env.action_spec.index_of("MOVE_E")
        env.step(move_e)
        # Box can't move into border wall
        assert env._box_positions[0] == old_box

    @pytest.mark.parametrize("cls", BOXOBAN_CLASSES)
    def test_cannot_push_box_into_box(self, cls: type) -> None:
        """Two boxes in a row: can't push the first into the second."""
        env = cls(max_turns=100)
        env.reset(0)
        s = env._grid_size

        env._player_pos = (1, s // 2)
        env._box_positions = [(2, s // 2), (3, s // 2)]
        env._target_positions = [(s - 2, 1), (s - 2, 2)]
        env._grid[s // 2][2] = "·"
        env._grid[s // 2][3] = "·"

        old_boxes = list(env._box_positions)
        move_e = env.action_spec.index_of("MOVE_E")
        env.step(move_e)
        # Neither box should have moved
        assert env._box_positions == old_boxes

    @pytest.mark.parametrize("cls", BOXOBAN_CLASSES)
    def test_action_spec(self, cls: type) -> None:
        env = cls()
        assert env.action_spec.n == 22
        assert env.noop_action_name == "WAIT"

    @pytest.mark.parametrize("cls", BOXOBAN_CLASSES)
    def test_max_turns_truncation(self, cls: type) -> None:
        env = cls(max_turns=3)
        env.reset(0)
        wait = env.action_spec.index_of("WAIT")
        for i in range(3):
            _, _, terminated, truncated, _ = env.step(wait)
            if i < 2:
                assert not truncated
            else:
                assert truncated
                assert not terminated

    @pytest.mark.parametrize("cls", BOXOBAN_CLASSES)
    def test_system_prompt(self, cls: type) -> None:
        env = cls()
        prompt = env.system_prompt()
        assert len(prompt) > 50
        assert "Sokoban" in prompt or "box" in prompt.lower()

    def test_medium_config(self) -> None:
        env = MiniHackBoxobanMediumEnv()
        assert env._grid_size == 7
        assert env._num_boxes == 2

    def test_hard_config(self) -> None:
        env = MiniHackBoxobanHardEnv()
        assert env._grid_size == 9
        assert env._num_boxes == 3

    def test_unfiltered_config(self) -> None:
        env = MiniHackBoxobanUnfilteredEnv()
        assert env._grid_size == 9
        assert env._num_boxes == 3

    @pytest.mark.parametrize("cls", BOXOBAN_CLASSES)
    def test_win_gives_reward(self, cls: type) -> None:
        """Manually place all boxes on targets and verify reward."""
        env = cls(max_turns=100)
        env.reset(0)
        s = env._grid_size

        # Set up: player at (1,1), one box at (3,3), target at (4,3)
        env._player_pos = (2, s // 2)
        env._box_positions = [(3, s // 2)]
        env._target_positions = [(4, s // 2)]
        env._grid[s // 2][3] = "·"
        env._grid[s // 2][4] = "·"

        move_e = env.action_spec.index_of("MOVE_E")
        _, reward, terminated, _, _ = env.step(move_e)
        assert terminated
        assert reward == 1.0

    @pytest.mark.parametrize("cls", BOXOBAN_CLASSES)
    @pytest.mark.parametrize("seed", range(5))
    def test_fuzz_random_actions(self, cls: type, seed: int) -> None:
        """Random actions should never raise exceptions."""
        import numpy as np

        env = cls(max_turns=50)
        env.reset(seed)
        rng = np.random.default_rng(seed)
        for _ in range(50):
            action = int(rng.integers(0, env.action_spec.n))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
