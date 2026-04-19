"""Tests for MiniHack KeyRoom environments."""

from __future__ import annotations

import pytest

from atlas_rl.envs.minihack.keyroom import (
    MiniHackKeyRoomDarkS5Env,
    MiniHackKeyRoomDarkS15Env,
    MiniHackKeyRoomS5Env,
    MiniHackKeyRoomS15Env,
)

KEYROOM_CLASSES = [
    MiniHackKeyRoomS5Env,
    MiniHackKeyRoomS15Env,
    MiniHackKeyRoomDarkS5Env,
    MiniHackKeyRoomDarkS15Env,
]
KEYROOM_IDS = [
    "atlas_rl/minihack-keyroom-s5-v0",
    "atlas_rl/minihack-keyroom-s15-v0",
    "atlas_rl/minihack-keyroom-dark-s5-v0",
    "atlas_rl/minihack-keyroom-dark-s15-v0",
]


class TestKeyRoomEnvs:
    """Tests for all KeyRoom variants."""

    @pytest.mark.parametrize(
        "cls,expected_id", zip(KEYROOM_CLASSES, KEYROOM_IDS, strict=True)
    )
    def test_env_id(self, cls: type, expected_id: str) -> None:
        env = cls()
        assert env.env_id() == expected_id

    @pytest.mark.parametrize("cls", KEYROOM_CLASSES)
    def test_reset_determinism(self, cls: type) -> None:
        e1, e2 = cls(), cls()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    @pytest.mark.parametrize("cls", KEYROOM_CLASSES)
    def test_reset_produces_valid_grid(self, cls: type) -> None:
        env = cls()
        obs_str, _ = env.reset(seed=0)
        assert "@" in obs_str
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str

    @pytest.mark.parametrize("cls", KEYROOM_CLASSES)
    def test_grid_rows_same_length(self, cls: type) -> None:
        env = cls()
        env.reset(seed=0)
        grid_obs = env.get_observation()
        lines = grid_obs.grid.split("\n")
        lengths = [len(line) for line in lines]
        assert len(set(lengths)) == 1, f"Unequal row lengths: {lengths}"

    @pytest.mark.parametrize("cls", KEYROOM_CLASSES)
    def test_door_exists_in_grid(self, cls: type) -> None:
        """Grid should contain a door (+) at reset."""
        env = cls()
        env.reset(seed=0)
        # Door might not be visible in dark variants from starting pos,
        # so check the internal grid directly
        flat = "".join("".join(row) for row in env._grid)
        assert "+" in flat, "Grid should contain a door (+)"

    def test_door_opens_on_move_into(self) -> None:
        """Moving into the door should open it."""
        env = MiniHackKeyRoomS5Env()
        env.reset(seed=0)

        # Find the door position
        door_pos = None
        for y in range(env._grid_h):
            for x in range(env._grid_w):
                if env._grid[y][x] == "+":
                    door_pos = (x, y)
                    break
            if door_pos:
                break
        assert door_pos is not None

        # Navigate player to be adjacent to the door
        # Door is at partition_x = w // 2, door_y = h // 2
        # Player starts at (1, h // 2) -- same y as door
        # So just move east until we hit the door
        move_e = env.action_spec.index_of("MOVE_E")
        for _ in range(env._grid_w):
            obs, reward, terminated, truncated, info = env.step(move_e)
            if terminated or truncated:
                break
            # Check if door was opened
            if env._grid[door_pos[1]][door_pos[0]] == ".":
                break

        # Door should be opened (changed to floor)
        assert env._grid[door_pos[1]][door_pos[0]] == ".", "Door should be opened"

    def test_can_reach_goal_through_door(self) -> None:
        """Agent should be able to reach stairs by going east through the door."""
        env = MiniHackKeyRoomS5Env(max_turns=100)
        env.reset(seed=0)
        move_e = env.action_spec.index_of("MOVE_E")
        reached = False
        for _ in range(100):
            _, reward, terminated, truncated, _ = env.step(move_e)
            if terminated:
                assert reward == 1.0
                reached = True
                break
            if truncated:
                break
        assert reached, "Agent should reach goal through door"

    def test_dark_variant_has_limited_visibility(self) -> None:
        env = MiniHackKeyRoomDarkS5Env()
        env.reset(seed=0)
        assert env._dark is True
        obs = env.get_observation()
        rendered = obs.render()
        assert " " in rendered, "Dark variant should have unseen spaces"

    def test_light_variant_is_fully_visible(self) -> None:
        env = MiniHackKeyRoomS5Env()
        env.reset(seed=0)
        assert env._dark is False

    @pytest.mark.parametrize("cls", KEYROOM_CLASSES)
    def test_action_spec(self, cls: type) -> None:
        env = cls()
        assert env.action_spec.n == 22
        assert env.noop_action_name == "WAIT"

    @pytest.mark.parametrize("cls", KEYROOM_CLASSES)
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

    @pytest.mark.parametrize("cls", KEYROOM_CLASSES)
    def test_system_prompt(self, cls: type) -> None:
        env = cls()
        prompt = env.system_prompt()
        assert len(prompt) > 50
