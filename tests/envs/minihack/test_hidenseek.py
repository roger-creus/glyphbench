"""Tests for MiniHack HideNSeek environments."""

from __future__ import annotations

import pytest

from atlas_rl.envs.minihack.hidenseek import (
    MiniHackHideNSeekBigEnv,
    MiniHackHideNSeekEnv,
    MiniHackHideNSeekLavaEnv,
    MiniHackHideNSeekMappedEnv,
)

HIDENSEEK_CLASSES = [
    MiniHackHideNSeekEnv,
    MiniHackHideNSeekMappedEnv,
    MiniHackHideNSeekLavaEnv,
    MiniHackHideNSeekBigEnv,
]
HIDENSEEK_IDS = [
    "atlas_rl/minihack-hidenseek-v0",
    "atlas_rl/minihack-hidenseek-mapped-v0",
    "atlas_rl/minihack-hidenseek-lava-v0",
    "atlas_rl/minihack-hidenseek-big-v0",
]


class TestHideNSeekEnvs:
    """Tests for all HideNSeek variants."""

    @pytest.mark.parametrize(
        "cls,expected_id", zip(HIDENSEEK_CLASSES, HIDENSEEK_IDS, strict=True)
    )
    def test_env_id(self, cls: type, expected_id: str) -> None:
        env = cls()
        assert env.env_id() == expected_id

    @pytest.mark.parametrize("cls", HIDENSEEK_CLASSES)
    def test_reset_determinism(self, cls: type) -> None:
        e1, e2 = cls(), cls()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("cls", HIDENSEEK_CLASSES)
    def test_reset_produces_valid_grid(self, cls: type) -> None:
        env = cls()
        obs_str, _ = env.reset(seed=0)
        assert "@" in obs_str
        assert ">" in obs_str
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str
        assert "[HUD]" in obs_str

    @pytest.mark.parametrize("cls", HIDENSEEK_CLASSES)
    def test_grid_rows_same_length(self, cls: type) -> None:
        env = cls()
        env.reset(seed=0)
        grid_obs = env.get_observation()
        lines = grid_obs.grid.split("\n")
        lengths = [len(line) for line in lines]
        assert len(set(lengths)) == 1, f"Unequal row lengths: {lengths}"

    @pytest.mark.parametrize("cls", HIDENSEEK_CLASSES)
    def test_has_walls(self, cls: type) -> None:
        """Grid should contain internal walls (#) forming corridors."""
        env = cls()
        env.reset(seed=0)
        wall_count = sum(
            1 for row in env._grid for cell in row if cell == "#"
        )
        assert wall_count > 0, "HideNSeek should have internal walls"

    @pytest.mark.parametrize("cls", HIDENSEEK_CLASSES)
    def test_has_creature(self, cls: type) -> None:
        env = cls()
        env.reset(seed=0)
        assert len(env._creatures) > 0, "HideNSeek should spawn a creature"

    @pytest.mark.parametrize("cls", HIDENSEEK_CLASSES)
    def test_random_rollout(self, cls: type) -> None:
        """Random rollout should not crash."""
        env = cls(max_turns=100)
        env.reset(seed=7)
        for _ in range(100):
            action = int(env.rng.integers(0, env.action_spec.n))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    @pytest.mark.parametrize("cls", HIDENSEEK_CLASSES)
    def test_action_spec(self, cls: type) -> None:
        env = cls()
        assert env.action_spec.n == 15
        assert env.noop_action_name == "WAIT"

    @pytest.mark.parametrize("cls", HIDENSEEK_CLASSES)
    def test_system_prompt(self, cls: type) -> None:
        env = cls()
        prompt = env.system_prompt()
        assert len(prompt) > 50
        assert "MOVE_N" in prompt

    def test_big_is_15x15(self) -> None:
        env = MiniHackHideNSeekBigEnv()
        env.reset(seed=0)
        assert env._grid_w == 15
        assert env._grid_h == 15

    def test_standard_is_9x9(self) -> None:
        env = MiniHackHideNSeekEnv()
        env.reset(seed=0)
        assert env._grid_w == 9
        assert env._grid_h == 9

    def test_lava_variant_has_lava(self) -> None:
        env = MiniHackHideNSeekLavaEnv()
        env.reset(seed=0)
        has_lava = any(cell == "}" for row in env._grid for cell in row)
        # Lava is random (30% chance per cell), check across multiple seeds
        found = False
        for s in range(10):
            env.reset(seed=s)
            if any(cell == "}" for row in env._grid for cell in row):
                found = True
                break
        assert found, "Lava variant should place lava in at least one seed"
