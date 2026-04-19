"""Tests for MiniHack Memento environments."""

from __future__ import annotations

import pytest

from atlas_rl.envs.minihack.memento import (
    MiniHackMementoF2Env,
    MiniHackMementoF4Env,
    MiniHackMementoHardEnv,
    MiniHackMementoShortEnv,
)

MEMENTO_CLASSES = [
    MiniHackMementoShortEnv,
    MiniHackMementoHardEnv,
    MiniHackMementoF2Env,
    MiniHackMementoF4Env,
]
MEMENTO_IDS = [
    "atlas_rl/minihack-memento-short-v0",
    "atlas_rl/minihack-memento-hard-v0",
    "atlas_rl/minihack-memento-f2-v0",
    "atlas_rl/minihack-memento-f4-v0",
]


class TestMementoEnvs:
    """Tests for all Memento variants."""

    @pytest.mark.parametrize(
        "cls,expected_id", zip(MEMENTO_CLASSES, MEMENTO_IDS, strict=True)
    )
    def test_env_id(self, cls: type, expected_id: str) -> None:
        env = cls()
        assert env.env_id() == expected_id

    @pytest.mark.parametrize("cls", MEMENTO_CLASSES)
    def test_reset_determinism(self, cls: type) -> None:
        e1, e2 = cls(), cls()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("cls", MEMENTO_CLASSES)
    def test_reset_produces_valid_grid(self, cls: type) -> None:
        env = cls()
        obs_str, _ = env.reset(seed=0)
        assert "@" in obs_str
        assert ">" in obs_str
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str
        assert "[HUD]" in obs_str

    @pytest.mark.parametrize("cls", MEMENTO_CLASSES)
    def test_grid_rows_same_length(self, cls: type) -> None:
        env = cls()
        env.reset(seed=0)
        grid_obs = env.get_observation()
        lines = grid_obs.grid.split("\n")
        lengths = [len(line) for line in lines]
        assert len(set(lengths)) == 1, f"Unequal row lengths: {lengths}"

    @pytest.mark.parametrize("cls", MEMENTO_CLASSES[:2])
    def test_single_floor_reach_goal(self, cls: type) -> None:
        """Navigate east to reach stairs (single-floor variants)."""
        env = cls(max_turns=500)
        env.reset(seed=0)
        move_e = env.action_spec.index_of("MOVE_E")
        reached = False
        for _ in range(500):
            _, reward, terminated, truncated, info = env.step(move_e)
            if terminated:
                assert reward == 1.0
                assert info.get("goal_reached")
                reached = True
                break
            if truncated:
                break
        assert reached, "Agent should reach goal by moving east"

    def test_multi_floor_f2(self) -> None:
        """F2 env requires reaching stairs twice."""
        env = MiniHackMementoF2Env(max_turns=1000)
        env.reset(seed=0)
        move_e = env.action_spec.index_of("MOVE_E")
        floors_seen: set[int] = {1}
        final_reward = 0.0
        for _ in range(1000):
            _, reward, terminated, truncated, info = env.step(move_e)
            floors_seen.add(info.get("floor", 1))
            if terminated:
                final_reward = reward
                break
            if truncated:
                break
        assert len(floors_seen) >= 2, f"Should visit 2 floors, saw {floors_seen}"
        assert final_reward == 1.0

    @pytest.mark.parametrize("cls", MEMENTO_CLASSES)
    def test_walls_block_movement(self, cls: type) -> None:
        env = cls()
        env.reset(seed=0)
        move_n = env.action_spec.index_of("MOVE_N")
        for _ in range(20):
            env.step(move_n)
        assert env._player_pos[1] >= 1

    @pytest.mark.parametrize("cls", MEMENTO_CLASSES)
    def test_action_spec(self, cls: type) -> None:
        env = cls()
        assert env.action_spec.n == 22
        assert env.noop_action_name == "WAIT"

    @pytest.mark.parametrize("cls", MEMENTO_CLASSES)
    def test_max_turns_truncation(self, cls: type) -> None:
        env = cls(max_turns=3)
        env.reset(seed=0)
        wait = env.action_spec.index_of("WAIT")
        for i in range(3):
            _, _, terminated, truncated, _ = env.step(wait)
            if i < 2:
                assert not truncated
            else:
                assert truncated
                assert not terminated

    @pytest.mark.parametrize("cls", MEMENTO_CLASSES)
    def test_system_prompt(self, cls: type) -> None:
        env = cls()
        prompt = env.system_prompt()
        assert len(prompt) > 50
        assert "MOVE_N" in prompt

    def test_short_config(self) -> None:
        env = MiniHackMementoShortEnv()
        assert env._num_rooms == 2
        assert env._num_floors == 1

    def test_hard_config(self) -> None:
        env = MiniHackMementoHardEnv()
        assert env._num_rooms == 4
        assert env._num_floors == 1

    def test_f2_config(self) -> None:
        env = MiniHackMementoF2Env()
        assert env._num_rooms == 3
        assert env._num_floors == 2

    def test_f4_config(self) -> None:
        env = MiniHackMementoF4Env()
        assert env._num_rooms == 3
        assert env._num_floors == 4

    @pytest.mark.parametrize("cls", MEMENTO_CLASSES)
    @pytest.mark.parametrize("seed", range(5))
    def test_fuzz_random_actions(self, cls: type, seed: int) -> None:
        """Random actions should never raise exceptions."""
        import numpy as np

        env = cls(max_turns=50)
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(50):
            action = int(rng.integers(0, env.action_spec.n))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
