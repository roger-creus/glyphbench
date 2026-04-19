"""Tests for MiniHack River environments."""

from __future__ import annotations

import pytest

from atlas_rl.envs.minihack.river import (
    MiniHackRiverEnv,
    MiniHackRiverLavaEnv,
    MiniHackRiverMonsterEnv,
    MiniHackRiverMonsterLavaEnv,
    MiniHackRiverNarrowEnv,
)

RIVER_CLASSES = [
    MiniHackRiverEnv,
    MiniHackRiverNarrowEnv,
    MiniHackRiverMonsterEnv,
    MiniHackRiverLavaEnv,
    MiniHackRiverMonsterLavaEnv,
]
RIVER_IDS = [
    "atlas_rl/minihack-river-v0",
    "atlas_rl/minihack-river-narrow-v0",
    "atlas_rl/minihack-river-monster-v0",
    "atlas_rl/minihack-river-lava-v0",
    "atlas_rl/minihack-river-monsterlava-v0",
]


class TestRiverEnvs:
    """Tests for all River variants."""

    @pytest.mark.parametrize(
        "cls,expected_id", zip(RIVER_CLASSES, RIVER_IDS, strict=True)
    )
    def test_env_id(self, cls: type, expected_id: str) -> None:
        env = cls()
        assert env.env_id() == expected_id

    @pytest.mark.parametrize("cls", RIVER_CLASSES)
    def test_reset_determinism(self, cls: type) -> None:
        e1, e2 = cls(), cls()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("cls", RIVER_CLASSES)
    def test_reset_produces_valid_grid(self, cls: type) -> None:
        env = cls()
        obs_str, _ = env.reset(seed=0)
        assert "@" in obs_str
        assert ">" in obs_str
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str
        assert "[HUD]" in obs_str

    @pytest.mark.parametrize("cls", RIVER_CLASSES)
    def test_grid_rows_same_length(self, cls: type) -> None:
        env = cls()
        env.reset(seed=0)
        grid_obs = env.get_observation()
        lines = grid_obs.grid.split("\n")
        lengths = [len(line) for line in lines]
        assert len(set(lengths)) == 1, f"Unequal row lengths: {lengths}"

    @pytest.mark.parametrize("cls", RIVER_CLASSES)
    def test_river_row_exists(self, cls: type) -> None:
        """The middle row of the grid should contain river tiles."""
        env = cls()
        env.reset(seed=0)
        h = env._grid_h
        river_y = h // 2
        river_row = env._grid[river_y]
        # At least some tiles should be river or stepping stones
        river_chars = {"~", "}", "."}
        inner = river_row[1:-1]
        assert all(c in river_chars for c in inner)

    @pytest.mark.parametrize("cls", RIVER_CLASSES)
    def test_random_rollout(self, cls: type) -> None:
        """Random rollout should not crash."""
        env = cls(max_turns=100)
        env.reset(seed=7)
        for _ in range(100):
            action = int(env.rng.integers(0, env.action_spec.n))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    @pytest.mark.parametrize("cls", RIVER_CLASSES)
    def test_action_spec(self, cls: type) -> None:
        env = cls()
        assert env.action_spec.n == 15
        assert env.noop_action_name == "WAIT"

    @pytest.mark.parametrize("cls", RIVER_CLASSES)
    def test_system_prompt(self, cls: type) -> None:
        env = cls()
        prompt = env.system_prompt()
        assert len(prompt) > 50
        assert "MOVE_N" in prompt

    def test_narrow_has_1_stone(self) -> None:
        env = MiniHackRiverNarrowEnv()
        assert env._num_stones == 1

    def test_lava_uses_lava_tiles(self) -> None:
        env = MiniHackRiverLavaEnv()
        env.reset(seed=0)
        river_y = env._grid_h // 2
        river_row = env._grid[river_y][1:-1]
        assert "}" in river_row, "Lava river should contain } tiles"

    def test_water_uses_water_tiles(self) -> None:
        env = MiniHackRiverEnv()
        env.reset(seed=0)
        river_y = env._grid_h // 2
        river_row = env._grid[river_y][1:-1]
        assert "~" in river_row, "Water river should contain ~ tiles"

    def test_monster_variant_has_creatures(self) -> None:
        env = MiniHackRiverMonsterEnv()
        env.reset(seed=0)
        assert len(env._creatures) > 0, "Monster variant should spawn creatures"

    def test_no_monster_variant_has_no_creatures(self) -> None:
        env = MiniHackRiverEnv()
        env.reset(seed=0)
        assert len(env._creatures) == 0, "Base river should have no creatures"
