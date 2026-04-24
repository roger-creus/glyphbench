"""Tests for MiniHack skill task environments."""

from __future__ import annotations

from glyphbench.core import make_env
import glyphbench.envs.minihack  # register envs


import pytest

import glyphbench  # noqa: F401

SKILL_ENVS = [
    "glyphbench/minihack-eat-v0",
    "glyphbench/minihack-eat-distract-v0",
    "glyphbench/minihack-pray-v0",
    "glyphbench/minihack-pray-distract-v0",
    "glyphbench/minihack-sink-v0",
    "glyphbench/minihack-sink-distract-v0",
    "glyphbench/minihack-read-v0",
    "glyphbench/minihack-read-distract-v0",
    "glyphbench/minihack-quaff-v0",
    "glyphbench/minihack-quaff-distract-v0",
    "glyphbench/minihack-wield-v0",
    "glyphbench/minihack-wield-distract-v0",
]


class TestSkillReset:
    """All skill envs reset cleanly and produce valid observations."""

    @pytest.mark.parametrize("env_id", SKILL_ENVS)
    def test_reset_and_step(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=50)
        obs, info = env.reset(42)
        assert isinstance(obs, str)
        assert len(obs) > 0
        assert "@" in obs
        # Take one step
        obs2, reward, terminated, truncated, info2 = env.step(8)  # WAIT
        assert isinstance(obs2, str)

    @pytest.mark.parametrize("env_id", SKILL_ENVS)
    def test_seed_determinism(self, env_id: str) -> None:
        for seed in [0, 42, 99]:
            e1 = make_env(env_id, max_turns=50)
            e2 = make_env(env_id, max_turns=50)
            o1, _ = e1.reset(seed)
            o2, _ = e2.reset(seed)
            assert o1 == o2
            for _ in range(5):
                a = 8  # WAIT
                o1, r1, t1, tr1, _ = e1.step(a)
                o2, r2, t2, tr2, _ = e2.step(a)
                assert o1 == o2
                assert r1 == r2
                if t1 or tr1:
                    break

    @pytest.mark.parametrize("env_id", SKILL_ENVS)
    def test_random_rollout(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=100)
        obs, _ = env.reset(7)
        for _ in range(100):
            action = int(env.rng.integers(0, env.action_spec.n))
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(obs, str)
            assert len(obs) > 0
            if terminated or truncated:
                break


class TestEatSpecific:
    """Eat-specific mechanics."""

    def test_starvation_kills(self) -> None:
        env = make_env("glyphbench/minihack-eat-v0", max_turns=200)
        env.reset(0)
        # Wait until starvation (hunger starts at 20)
        for _ in range(25):
            obs, reward, terminated, truncated, info = env.step(8)  # WAIT
            if terminated:
                assert reward == -1.0
                assert info.get("cause_of_death") == "starvation"
                return
        pytest.fail("Agent did not starve within 25 turns")

    def test_eating_restores_hunger(self) -> None:
        env = make_env("glyphbench/minihack-eat-v0", max_turns=200)
        env.reset(0)
        from glyphbench.core.base_env import BaseGlyphEnv

        unwrapped: BaseGlyphEnv = env  # type: ignore[assignment]
        from glyphbench.envs.minihack.skill_eat import _EatBase

        assert isinstance(unwrapped, _EatBase)
        # Navigate to food at (3,3) from (1,1): move E twice then S twice
        move_e = unwrapped.action_spec.index_of("MOVE_E")
        move_s = unwrapped.action_spec.index_of("MOVE_S")
        pickup = unwrapped.action_spec.index_of("PICKUP")
        eat = unwrapped.action_spec.index_of("EAT")

        env.step(move_e)
        env.step(move_e)
        env.step(move_s)
        env.step(move_s)

        # Should be at (3,3) now -- pick up food and eat
        env.step(pickup)
        assert len(unwrapped._inventory) == 1
        hunger_before = unwrapped._hunger
        env.step(eat)
        assert unwrapped._hunger > hunger_before


class TestPraySpecific:
    """Pray-specific mechanics."""

    def test_pray_heals(self) -> None:
        env = make_env("glyphbench/minihack-pray-v0", max_turns=50)
        env.reset(0)
        from glyphbench.core.base_env import BaseGlyphEnv

        unwrapped: BaseGlyphEnv = env  # type: ignore[assignment]
        from glyphbench.envs.minihack.skill_pray import _PrayBase

        assert isinstance(unwrapped, _PrayBase)
        assert unwrapped._player_hp == 3  # starts low
        pray = unwrapped.action_spec.index_of("PRAY")
        env.step(pray)
        assert unwrapped._player_hp == unwrapped._player_max_hp


class TestReadSpecific:
    """Read-specific mechanics."""

    def test_read_teleport_to_stairs(self) -> None:
        env = make_env("glyphbench/minihack-read-v0", max_turns=50)
        env.reset(0)
        from glyphbench.core.base_env import BaseGlyphEnv

        unwrapped: BaseGlyphEnv = env  # type: ignore[assignment]
        from glyphbench.envs.minihack.skill_read import _ReadBase

        assert isinstance(unwrapped, _ReadBase)
        # Navigate to scroll at (2,2) from (1,1): move E then S
        move_e = unwrapped.action_spec.index_of("MOVE_E")
        move_s = unwrapped.action_spec.index_of("MOVE_S")
        pickup = unwrapped.action_spec.index_of("PICKUP")
        read = unwrapped.action_spec.index_of("READ")

        env.step(move_e)
        env.step(move_s)
        # At (2,2), pick up scroll
        env.step(pickup)
        assert len(unwrapped._inventory) == 1
        # Read it -- should teleport to stairs and terminate
        obs, reward, terminated, truncated, info = env.step(read)
        assert terminated
        assert reward == 1.0
        assert info.get("goal_reached") is True


class TestQuaffSpecific:
    """Quaff-specific mechanics."""

    def test_quaff_heals(self) -> None:
        env = make_env("glyphbench/minihack-quaff-v0", max_turns=50)
        env.reset(0)
        from glyphbench.core.base_env import BaseGlyphEnv

        unwrapped: BaseGlyphEnv = env  # type: ignore[assignment]
        from glyphbench.envs.minihack.skill_quaff import _QuaffBase

        assert isinstance(unwrapped, _QuaffBase)
        assert unwrapped._player_hp == 3
        # Navigate to potion at (2,2) from (1,1): move E then S
        move_e = unwrapped.action_spec.index_of("MOVE_E")
        move_s = unwrapped.action_spec.index_of("MOVE_S")
        pickup = unwrapped.action_spec.index_of("PICKUP")
        quaff = unwrapped.action_spec.index_of("QUAFF")

        env.step(move_e)
        env.step(move_s)
        env.step(pickup)
        assert len(unwrapped._inventory) == 1
        hp_before_quaff = unwrapped._player_hp
        env.step(quaff)
        # Quaffing heals to max HP, but monster may attack same turn,
        # so just check HP is significantly higher than the 3 we started with
        assert unwrapped._player_hp > hp_before_quaff


class TestWieldSpecific:
    """Wield-specific mechanics."""

    def test_wield_equips_weapon(self) -> None:
        env = make_env("glyphbench/minihack-wield-v0", max_turns=50)
        env.reset(0)
        from glyphbench.core.base_env import BaseGlyphEnv

        unwrapped: BaseGlyphEnv = env  # type: ignore[assignment]
        from glyphbench.envs.minihack.skill_wield import _WieldBase

        assert isinstance(unwrapped, _WieldBase)
        # Weapon at (2,1), player at (1,1): move E
        move_e = unwrapped.action_spec.index_of("MOVE_E")
        pickup = unwrapped.action_spec.index_of("PICKUP")
        wield = unwrapped.action_spec.index_of("WIELD")

        env.step(move_e)
        env.step(pickup)
        assert len(unwrapped._inventory) == 1
        assert unwrapped._wielding is None
        env.step(wield)
        assert unwrapped._wielding is not None
        assert unwrapped._wielding.name == "long sword"


class TestSinkSpecific:
    """Sink-specific mechanics."""

    def test_sink_terrain_visible(self) -> None:
        env = make_env("glyphbench/minihack-sink-v0", max_turns=50)
        obs, _ = env.reset(0)
        assert "{" in obs

    def test_sink_walkable(self) -> None:
        env = make_env("glyphbench/minihack-sink-v0", max_turns=50)
        env.reset(0)
        from glyphbench.core.base_env import BaseGlyphEnv

        unwrapped: BaseGlyphEnv = env  # type: ignore[assignment]
        from glyphbench.envs.minihack.skill_sink import _SinkBase

        assert isinstance(unwrapped, _SinkBase)
        # Navigate to sink at (3,3)
        move_e = unwrapped.action_spec.index_of("MOVE_E")
        move_s = unwrapped.action_spec.index_of("MOVE_S")

        env.step(move_e)
        env.step(move_e)
        env.step(move_s)
        env.step(move_s)
        assert unwrapped._player_pos == (3, 3)


class TestDistractVariants:
    """Distract variants have extra items on the floor."""

    @pytest.mark.parametrize(
        "env_id",
        [eid for eid in SKILL_ENVS if "distract" in eid],
    )
    def test_distract_has_extra_items(self, env_id: str) -> None:
        env = make_env(env_id, max_turns=50)
        env.reset(0)
        from glyphbench.core.base_env import BaseGlyphEnv

        unwrapped: BaseGlyphEnv = env  # type: ignore[assignment]
        from glyphbench.envs.minihack.base import MiniHackBase

        assert isinstance(unwrapped, MiniHackBase)
        # Distract variants should have at least 2 floor item positions
        assert len(unwrapped._floor_items) >= 2
