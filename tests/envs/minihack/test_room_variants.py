"""Parametrized tests for all MiniHack Room variant environments."""

from __future__ import annotations

import pytest

from atlas_rl.envs.minihack.room import (
    MiniHackRoom5x5Env,
    MiniHackRoom15x15Env,
    MiniHackRoomDark5x5Env,
    MiniHackRoomDark15x15Env,
    MiniHackRoomMonster5x5Env,
    MiniHackRoomMonster15x15Env,
    MiniHackRoomRandom5x5Env,
    MiniHackRoomRandom15x15Env,
    MiniHackRoomTrap5x5Env,
    MiniHackRoomTrap15x15Env,
    MiniHackRoomUltimate5x5Env,
    MiniHackRoomUltimate15x15Env,
)

# (class, expected_env_id, grid_size, has_monsters, has_traps, is_dark)
VARIANTS = [
    (MiniHackRoom5x5Env, "atlas_rl/minihack-room-5x5-v0", 7, False, False, False),
    (MiniHackRoom15x15Env, "atlas_rl/minihack-room-15x15-v0", 17, False, False, False),
    (MiniHackRoomRandom5x5Env, "atlas_rl/minihack-room-random-5x5-v0", 7, False, False, False),
    (MiniHackRoomRandom15x15Env, "atlas_rl/minihack-room-random-15x15-v0", 17, False, False, False),
    (MiniHackRoomDark5x5Env, "atlas_rl/minihack-room-dark-5x5-v0", 7, False, False, True),
    (MiniHackRoomDark15x15Env, "atlas_rl/minihack-room-dark-15x15-v0", 17, False, False, True),
    (MiniHackRoomMonster5x5Env, "atlas_rl/minihack-room-monster-5x5-v0", 7, True, False, False),
    (MiniHackRoomMonster15x15Env, "atlas_rl/minihack-room-monster-15x15-v0", 17, True, False, False),
    (MiniHackRoomTrap5x5Env, "atlas_rl/minihack-room-trap-5x5-v0", 7, False, True, False),
    (MiniHackRoomTrap15x15Env, "atlas_rl/minihack-room-trap-15x15-v0", 17, False, True, False),
    (MiniHackRoomUltimate5x5Env, "atlas_rl/minihack-room-ultimate-5x5-v0", 7, True, True, True),
    (MiniHackRoomUltimate15x15Env, "atlas_rl/minihack-room-ultimate-15x15-v0", 17, True, True, True),
]

VARIANT_IDS = [v[1].split("/")[1] for v in VARIANTS]


@pytest.fixture(params=VARIANTS, ids=VARIANT_IDS)
def variant(request: pytest.FixtureRequest):  # noqa: ANN201
    return request.param


class TestRoomVariants:
    """Shared tests across every Room variant."""

    def test_env_id(self, variant) -> None:  # noqa: ANN001
        cls, expected_id, *_ = variant
        env = cls(max_turns=50)
        assert env.env_id() == expected_id

    def test_action_space(self, variant) -> None:  # noqa: ANN001
        cls, *_ = variant
        env = cls(max_turns=50)
        assert env.action_spec.n == 22

    def test_reset_determinism(self, variant) -> None:  # noqa: ANN001
        cls, *_ = variant
        e1 = cls(max_turns=50)
        e2 = cls(max_turns=50)
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    def test_step_determinism(self, variant) -> None:  # noqa: ANN001
        cls, *_ = variant
        e1 = cls(max_turns=50)
        e2 = cls(max_turns=50)
        e1.reset(seed=7)
        e2.reset(seed=7)
        wait = e1.action_spec.index_of("WAIT")
        for _ in range(5):
            o1, r1, t1, tr1, _ = e1.step(wait)
            o2, r2, t2, tr2, _ = e2.step(wait)
            assert o1 == o2
            assert r1 == r2
            if t1 or tr1:
                break

    def test_observation_contract(self, variant) -> None:  # noqa: ANN001
        cls, *_ = variant
        env = cls(max_turns=50)
        obs_str, _ = env.reset(seed=0)
        assert isinstance(obs_str, str)
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str
        assert "[HUD]" in obs_str

    def test_grid_dimensions(self, variant) -> None:  # noqa: ANN001
        cls, _, grid_size, *_ = variant
        env = cls(max_turns=50)
        env.reset(seed=0)
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        assert len(grid_lines) == grid_size
        for line in grid_lines:
            assert len(line) == grid_size

    def test_player_and_goal_present(self, variant) -> None:  # noqa: ANN001
        cls, _, _, _, _, is_dark = variant
        env = cls(max_turns=50)
        env.reset(seed=0)
        grid_obs = env.get_observation()
        assert "@" in grid_obs.grid
        # In dark rooms the goal may be outside the vision radius
        if not is_dark:
            assert ">" in grid_obs.grid
        # But the goal must exist on the internal grid regardless
        assert env._goal_pos is not None

    def test_system_prompt(self, variant) -> None:  # noqa: ANN001
        cls, *_ = variant
        env = cls(max_turns=50)
        prompt = env.system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_max_turns_truncation(self, variant) -> None:  # noqa: ANN001
        cls, *_ = variant
        env = cls(max_turns=3)
        env.reset(seed=0)
        wait = env.action_spec.index_of("WAIT")
        for i in range(3):
            _, _, terminated, truncated, _ = env.step(wait)
            if terminated:
                break
            if i == 2:
                assert truncated

    def test_random_rollout_no_crash(self, variant) -> None:  # noqa: ANN001
        cls, *_ = variant
        env = cls(max_turns=100)
        for seed in (0, 1, 42):
            env.reset(seed=seed)
            for _ in range(100):
                a = int(env.rng.integers(0, env.action_spec.n))
                obs, r, t, tr, info = env.step(a)
                assert isinstance(obs, str)
                assert len(obs) > 0
                if t or tr:
                    break


class TestRoomMonsterSpecific:
    """Tests specific to monster-containing rooms."""

    @pytest.mark.parametrize(
        "cls", [MiniHackRoomMonster5x5Env, MiniHackRoomMonster15x15Env]
    )
    def test_monsters_present(self, cls: type) -> None:
        env = cls(max_turns=50)
        env.reset(seed=0)
        assert len(env._creatures) > 0
        for c in env._creatures:
            assert c.hp > 0

    @pytest.mark.parametrize(
        "cls,min_n,max_n",
        [
            (MiniHackRoomMonster5x5Env, 1, 2),
            (MiniHackRoomMonster15x15Env, 3, 5),
        ],
    )
    def test_monster_count_range(
        self, cls: type, min_n: int, max_n: int
    ) -> None:
        counts: set[int] = set()
        for seed in range(20):
            env = cls(max_turns=50)
            env.reset(seed=seed)
            counts.add(len(env._creatures))
        assert all(min_n <= c <= max_n for c in counts)


class TestRoomTrapSpecific:
    """Tests specific to trap-containing rooms."""

    @pytest.mark.parametrize(
        "cls", [MiniHackRoomTrap5x5Env, MiniHackRoomTrap15x15Env]
    )
    def test_traps_present(self, cls: type) -> None:
        env = cls(max_turns=50)
        env.reset(seed=0)
        grid_obs = env.get_observation()
        assert "^" in grid_obs.grid

    @pytest.mark.parametrize(
        "cls,min_n,max_n",
        [
            (MiniHackRoomTrap5x5Env, 2, 3),
            (MiniHackRoomTrap15x15Env, 5, 8),
        ],
    )
    def test_trap_count_range(
        self, cls: type, min_n: int, max_n: int
    ) -> None:
        counts: set[int] = set()
        for seed in range(20):
            env = cls(max_turns=50)
            env.reset(seed=seed)
            grid_obs = env.get_observation()
            n_traps = grid_obs.grid.count("^")
            counts.add(n_traps)
        assert all(min_n <= c <= max_n for c in counts)


class TestRoomDarkSpecific:
    """Tests specific to dark rooms."""

    @pytest.mark.parametrize(
        "cls", [MiniHackRoomDark5x5Env, MiniHackRoomDark15x15Env]
    )
    def test_dark_flag_set(self, cls: type) -> None:
        env = cls(max_turns=50)
        env.reset(seed=0)
        assert env._dark is True

    @pytest.mark.parametrize(
        "cls,grid_size",
        [
            (MiniHackRoomDark5x5Env, 7),
            (MiniHackRoomDark15x15Env, 17),
        ],
    )
    def test_limited_vision(self, cls: type, grid_size: int) -> None:
        """In dark rooms, most cells should be spaces (unseen)."""
        env = cls(max_turns=50)
        env.reset(seed=0)
        grid_obs = env.get_observation()
        space_count = grid_obs.grid.count(" ")
        total = grid_size * grid_size
        # Most cells should be dark (space), at least half for 15x15
        if grid_size > 10:
            assert space_count > total // 2


class TestRoomUltimateSpecific:
    """Tests for ultimate rooms (monsters + traps + dark)."""

    @pytest.mark.parametrize(
        "cls",
        [MiniHackRoomUltimate5x5Env, MiniHackRoomUltimate15x15Env],
    )
    def test_all_hazards_present(self, cls: type) -> None:
        env = cls(max_turns=50)
        env.reset(seed=0)
        assert env._dark is True
        assert len(env._creatures) > 0
        # Traps may not be visible (dark room), but they exist on the grid
        trap_count = sum(
            1
            for row in env._grid
            for cell in row
            if cell == "^"
        )
        assert trap_count > 0
