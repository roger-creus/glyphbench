"""Tests for MiniHack Wand of Death environments."""

from __future__ import annotations

import pytest

from atlas_rl.envs.minihack.skill_wod import (
    MiniHackWoDEasyEnv,
    MiniHackWoDHardEnv,
    MiniHackWoDMediumEnv,
    MiniHackWoDProEnv,
)

WOD_CLASSES = [
    MiniHackWoDEasyEnv,
    MiniHackWoDMediumEnv,
    MiniHackWoDHardEnv,
    MiniHackWoDProEnv,
]
WOD_IDS = [
    "atlas_rl/minihack-wod-easy-v0",
    "atlas_rl/minihack-wod-medium-v0",
    "atlas_rl/minihack-wod-hard-v0",
    "atlas_rl/minihack-wod-pro-v0",
]


class TestWoDEnvs:
    """Tests for all WoD variants."""

    @pytest.mark.parametrize(
        "cls,expected_id", zip(WOD_CLASSES, WOD_IDS, strict=True)
    )
    def test_env_id(self, cls: type, expected_id: str) -> None:
        env = cls()
        assert env.env_id() == expected_id

    @pytest.mark.parametrize("cls", WOD_CLASSES)
    def test_reset_determinism(self, cls: type) -> None:
        e1, e2 = cls(), cls()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("cls", WOD_CLASSES)
    def test_reset_produces_valid_grid(self, cls: type) -> None:
        env = cls()
        obs_str, _ = env.reset(seed=0)
        assert "@" in obs_str
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str
        assert "[HUD]" in obs_str

    @pytest.mark.parametrize("cls", WOD_CLASSES)
    def test_grid_rows_same_length(self, cls: type) -> None:
        env = cls()
        env.reset(seed=0)
        grid_obs = env.get_observation()
        lines = grid_obs.grid.split("\n")
        lengths = [len(line) for line in lines]
        assert len(set(lengths)) == 1, f"Unequal row lengths: {lengths}"

    @pytest.mark.parametrize("cls", WOD_CLASSES)
    def test_has_monsters(self, cls: type) -> None:
        env = cls()
        env.reset(seed=0)
        assert len(env._creatures) > 0, "WoD should spawn monsters"

    @pytest.mark.parametrize("cls", WOD_CLASSES)
    def test_random_rollout(self, cls: type) -> None:
        """Random rollout should not crash."""
        env = cls(max_turns=100)
        env.reset(seed=7)
        for _ in range(100):
            action = int(env.rng.integers(0, env.action_spec.n))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    @pytest.mark.parametrize("cls", WOD_CLASSES)
    def test_action_spec(self, cls: type) -> None:
        env = cls()
        assert env.action_spec.n == 22
        assert env.noop_action_name == "WAIT"

    @pytest.mark.parametrize("cls", WOD_CLASSES)
    def test_system_prompt(self, cls: type) -> None:
        env = cls()
        prompt = env.system_prompt()
        assert len(prompt) > 50
        assert "MOVE_N" in prompt

    def test_easy_has_1_monster(self) -> None:
        env = MiniHackWoDEasyEnv()
        env.reset(seed=0)
        assert len(env._creatures) == 1

    def test_medium_has_2_monsters(self) -> None:
        env = MiniHackWoDMediumEnv()
        env.reset(seed=0)
        assert len(env._creatures) == 2

    def test_hard_has_3_monsters(self) -> None:
        env = MiniHackWoDHardEnv()
        env.reset(seed=0)
        assert len(env._creatures) == 3

    def test_pro_has_4_monsters(self) -> None:
        env = MiniHackWoDProEnv()
        env.reset(seed=0)
        assert len(env._creatures) == 4

    def test_pro_is_dark(self) -> None:
        env = MiniHackWoDProEnv()
        env.reset(seed=0)
        assert env._dark is True

    def test_easy_not_dark(self) -> None:
        env = MiniHackWoDEasyEnv()
        env.reset(seed=0)
        assert env._dark is False

    def test_hard_has_distractors(self) -> None:
        env = MiniHackWoDHardEnv()
        env.reset(seed=0)
        # Should have wand of death + 2 distractors on floor
        all_floor_items = [
            item
            for items in env._floor_items.values()
            for item in items
        ]
        wand_names = [i.name for i in all_floor_items if i.item_type == "wand"]
        assert "wand of death" in wand_names
        assert "wand of fire" in wand_names
        assert "wand of cold" in wand_names

    def test_zap_wand_kills_monster(self) -> None:
        """Picking up wand of death and zapping should kill a monster."""
        env = MiniHackWoDEasyEnv(max_turns=200)
        env.reset(seed=0)

        # Give player the wand directly for deterministic test
        from atlas_rl.envs.minihack.items import WAND_DEATH

        env._inventory.append(WAND_DEATH)
        initial_monsters = len(env._creatures)

        zap = env.action_spec.index_of("ZAP")
        env.step(zap)

        # Monster should be killed (unless player died from monster attacks)
        if env._player_hp > 0:
            assert len(env._creatures) < initial_monsters

    def test_zap_distractor_wand_no_kill(self) -> None:
        """Zapping a distractor wand should not kill a monster."""
        env = MiniHackWoDEasyEnv(max_turns=200)
        env.reset(seed=0)

        from atlas_rl.envs.minihack.items import WAND_FIRE

        env._inventory.append(WAND_FIRE)
        initial_monsters = len(env._creatures)

        zap = env.action_spec.index_of("ZAP")
        env.step(zap)

        # Monster count unchanged (no wand of death effect)
        alive = [c for c in env._creatures if c.hp > 0]
        assert len(alive) == initial_monsters
