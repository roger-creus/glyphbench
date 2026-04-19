"""Tests for MiniHack LavaCross environments."""

from __future__ import annotations

import pytest

from atlas_rl.envs.minihack.skill_lavacross import (
    MiniHackLavaCrossFullEnv,
    MiniHackLavaCrossLevitateEnv,
    MiniHackLavaCrossPotionInvEnv,
    MiniHackLavaCrossRingInvEnv,
)

LAVACROSS_CLASSES = [
    MiniHackLavaCrossFullEnv,
    MiniHackLavaCrossLevitateEnv,
    MiniHackLavaCrossPotionInvEnv,
    MiniHackLavaCrossRingInvEnv,
]
LAVACROSS_IDS = [
    "atlas_rl/minihack-lavacross-full-v0",
    "atlas_rl/minihack-lavacross-levitate-v0",
    "atlas_rl/minihack-lavacross-levitate-potion-inv-v0",
    "atlas_rl/minihack-lavacross-levitate-ring-inv-v0",
]


class TestLavaCrossEnvs:
    """Tests for all LavaCross variants."""

    @pytest.mark.parametrize(
        "cls,expected_id", zip(LAVACROSS_CLASSES, LAVACROSS_IDS, strict=True)
    )
    def test_env_id(self, cls: type, expected_id: str) -> None:
        env = cls()
        assert env.env_id() == expected_id

    @pytest.mark.parametrize("cls", LAVACROSS_CLASSES)
    def test_reset_determinism(self, cls: type) -> None:
        e1, e2 = cls(), cls()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("cls", LAVACROSS_CLASSES)
    def test_reset_produces_valid_grid(self, cls: type) -> None:
        env = cls()
        obs_str, _ = env.reset(seed=0)
        assert "@" in obs_str
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str
        assert "[HUD]" in obs_str

    @pytest.mark.parametrize("cls", LAVACROSS_CLASSES)
    def test_grid_rows_same_length(self, cls: type) -> None:
        env = cls()
        env.reset(seed=0)
        grid_obs = env.get_observation()
        lines = grid_obs.grid.split("\n")
        lengths = [len(line) for line in lines]
        assert len(set(lengths)) == 1, f"Unequal row lengths: {lengths}"

    @pytest.mark.parametrize("cls", LAVACROSS_CLASSES)
    def test_has_lava(self, cls: type) -> None:
        env = cls()
        env.reset(seed=0)
        lava_count = sum(
            1
            for y in range(env._grid_h)
            for x in range(env._grid_w)
            if env._grid[y][x] == "}"
        )
        assert lava_count > 0, "LavaCross should have lava tiles"

    @pytest.mark.parametrize("cls", LAVACROSS_CLASSES)
    def test_random_rollout(self, cls: type) -> None:
        """Random rollout should not crash."""
        env = cls(max_turns=100)
        env.reset(seed=7)
        for _ in range(100):
            action = int(env.rng.integers(0, env.action_spec.n))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    @pytest.mark.parametrize("cls", LAVACROSS_CLASSES)
    def test_action_spec(self, cls: type) -> None:
        env = cls()
        assert env.action_spec.n == 22
        assert env.noop_action_name == "WAIT"

    @pytest.mark.parametrize("cls", LAVACROSS_CLASSES)
    def test_system_prompt(self, cls: type) -> None:
        env = cls()
        prompt = env.system_prompt()
        assert len(prompt) > 50
        assert "lava" in prompt.lower()

    def test_full_has_both_items(self) -> None:
        env = MiniHackLavaCrossFullEnv()
        env.reset(seed=0)
        all_floor_items = [
            item
            for items in env._floor_items.values()
            for item in items
        ]
        names = [i.name for i in all_floor_items]
        assert "potion of levitation" in names
        assert "ring of levitation" in names

    def test_levitate_only_potion(self) -> None:
        env = MiniHackLavaCrossLevitateEnv()
        env.reset(seed=0)
        all_floor_items = [
            item
            for items in env._floor_items.values()
            for item in items
        ]
        names = [i.name for i in all_floor_items]
        assert "potion of levitation" in names
        assert "ring of levitation" not in names

    def test_potion_inv_starts_in_inventory(self) -> None:
        env = MiniHackLavaCrossPotionInvEnv()
        env.reset(seed=0)
        inv_names = [i.name for i in env._inventory]
        assert "potion of levitation" in inv_names

    def test_ring_inv_starts_in_inventory(self) -> None:
        env = MiniHackLavaCrossRingInvEnv()
        env.reset(seed=0)
        inv_names = [i.name for i in env._inventory]
        assert "ring of levitation" in inv_names

    def test_stepping_on_lava_kills(self) -> None:
        """Without levitation, stepping on lava should kill the player."""
        env = MiniHackLavaCrossFullEnv(max_turns=200)
        env.reset(seed=0)
        # Move player to adjacent to lava and step onto it
        env._player_pos = (3, 3)
        move_e = env.action_spec.index_of("MOVE_E")
        _, reward, terminated, _, _ = env.step(move_e)
        assert terminated
        assert reward == -1.0

    def test_levitation_prevents_lava_death(self) -> None:
        """With levitation active, stepping on lava should be safe."""
        env = MiniHackLavaCrossPotionInvEnv(max_turns=200)
        env.reset(seed=0)

        # Quaff potion to activate levitation
        quaff = env.action_spec.index_of("QUAFF")
        env.step(quaff)
        assert env._levitating_turns > 0

        # Position player adjacent to lava and step onto it
        env._player_pos = (3, 3)
        move_e = env.action_spec.index_of("MOVE_E")
        _, _, terminated, _, _ = env.step(move_e)
        assert not terminated
        assert env._player_hp > 0
        assert env._player_pos == (4, 3)

    def test_levitation_wears_off(self) -> None:
        """Levitation should decrement each turn and eventually expire."""
        env = MiniHackLavaCrossPotionInvEnv(max_turns=200)
        env.reset(seed=0)

        quaff = env.action_spec.index_of("QUAFF")
        env.step(quaff)
        initial_turns = env._levitating_turns

        wait = env.action_spec.index_of("WAIT")
        env.step(wait)
        assert env._levitating_turns == initial_turns - 1

    def test_lava_restored_after_levitating_step(self) -> None:
        """Lava tiles should be restored after a levitating step."""
        env = MiniHackLavaCrossPotionInvEnv(max_turns=200)
        env.reset(seed=0)

        quaff = env.action_spec.index_of("QUAFF")
        env.step(quaff)

        env._player_pos = (3, 3)
        move_e = env.action_spec.index_of("MOVE_E")
        env.step(move_e)

        # Lava tiles should still be lava in the grid
        lava_count = sum(
            1
            for y in range(env._grid_h)
            for x in range(env._grid_w)
            if env._grid[y][x] == "}"
        )
        assert lava_count > 0, "Lava tiles should be restored after stepping"
