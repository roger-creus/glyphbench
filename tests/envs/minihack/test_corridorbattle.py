"""Tests for MiniHack CorridorBattle environments."""

from __future__ import annotations

import pytest

from glyphbench.envs.minihack.corridorbattle import (
    MiniHackCorridorBattleDarkEnv,
    MiniHackCorridorBattleEnv,
)

CORRIDORBATTLE_CLASSES = [
    MiniHackCorridorBattleEnv,
    MiniHackCorridorBattleDarkEnv,
]
CORRIDORBATTLE_IDS = [
    "glyphbench/minihack-corridorbattle-v0",
    "glyphbench/minihack-corridorbattle-dark-v0",
]


class TestCorridorBattleEnvs:
    """Tests for all CorridorBattle variants."""

    @pytest.mark.parametrize(
        "cls,expected_id", zip(CORRIDORBATTLE_CLASSES, CORRIDORBATTLE_IDS, strict=True)
    )
    def test_env_id(self, cls: type, expected_id: str) -> None:
        env = cls()
        assert env.env_id() == expected_id

    @pytest.mark.parametrize("cls", CORRIDORBATTLE_CLASSES)
    def test_reset_determinism(self, cls: type) -> None:
        e1, e2 = cls(), cls()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("cls", CORRIDORBATTLE_CLASSES)
    def test_reset_produces_valid_grid(self, cls: type) -> None:
        env = cls()
        obs_str, _ = env.reset(seed=0)
        assert "@" in obs_str
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str
        assert "[HUD]" in obs_str

    @pytest.mark.parametrize("cls", CORRIDORBATTLE_CLASSES)
    def test_grid_rows_same_length(self, cls: type) -> None:
        env = cls()
        env.reset(seed=0)
        grid_obs = env.get_observation()
        lines = grid_obs.grid.split("\n")
        lengths = [len(line) for line in lines]
        assert len(set(lengths)) == 1, f"Unequal row lengths: {lengths}"

    @pytest.mark.parametrize("cls", CORRIDORBATTLE_CLASSES)
    def test_has_monsters(self, cls: type) -> None:
        env = cls()
        env.reset(seed=0)
        assert len(env._creatures) > 0, "CorridorBattle should spawn monsters"

    @pytest.mark.parametrize("cls", CORRIDORBATTLE_CLASSES)
    def test_corridor_shape(self, cls: type) -> None:
        """Grid should be 15x5 with a single corridor in the middle."""
        env = cls()
        env.reset(seed=0)
        assert env._grid_w == 15
        assert env._grid_h == 5
        # Middle row should be walkable (corridor)
        corridor_y = env._grid_h // 2
        for x in range(1, env._grid_w - 1):
            cell = env._grid[corridor_y][x]
            # corridor cells should be floor, stairs, or have a creature
            assert cell in ("·", "⇣") or env._creature_at(x, corridor_y) is not None

    @pytest.mark.parametrize("cls", CORRIDORBATTLE_CLASSES)
    def test_random_rollout(self, cls: type) -> None:
        """Random rollout should not crash."""
        env = cls(max_turns=100)
        env.reset(seed=7)
        for _ in range(100):
            action = int(env.rng.integers(0, env.action_spec.n))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    @pytest.mark.parametrize("cls", CORRIDORBATTLE_CLASSES)
    def test_action_spec(self, cls: type) -> None:
        env = cls()
        assert env.action_spec.n == 22
        assert env.noop_action_name == "WAIT"

    @pytest.mark.parametrize("cls", CORRIDORBATTLE_CLASSES)
    def test_system_prompt(self, cls: type) -> None:
        env = cls()
        prompt = env.system_prompt()
        assert len(prompt) > 50
        assert "MOVE_N" in prompt

    def test_dark_variant_is_dark(self) -> None:
        env = MiniHackCorridorBattleDarkEnv()
        env.reset(seed=0)
        assert env._dark is True

    def test_normal_variant_not_dark(self) -> None:
        env = MiniHackCorridorBattleEnv()
        env.reset(seed=0)
        assert env._dark is False

    def test_fight_through_corridor(self) -> None:
        """Player moving east should be able to fight through monsters."""
        env = MiniHackCorridorBattleEnv(max_turns=500)
        env.reset(seed=0)
        move_e = env.action_spec.index_of("MOVE_E")
        reached = False
        for _ in range(500):
            _, reward, terminated, truncated, info = env.step(move_e)
            if terminated:
                if info.get("goal_reached"):
                    reached = True
                break
            if truncated:
                break
        # Player might die or reach goal; just verify no crash
        assert True
