"""Tests for MiniHackBase shared functionality."""

from __future__ import annotations

from atlas_rl.envs.minihack.creatures import KOBOLD


class TestMiniHackBase:
    def _make_env(
        self,
        width: int = 7,
        height: int = 7,
        max_turns: int = 100,
        dark: bool = False,
        monsters: bool = False,
    ):  # type: ignore[no-untyped-def]
        from atlas_rl.envs.minihack.base import MiniHackBase

        class TestEnv(MiniHackBase):
            def env_id(self) -> str:
                return "atlas_rl/test-minihack-v0"

            def _generate_level(self, seed: int) -> None:
                self._init_grid(width, height)
                self._place_player(1, 1)
                self._place_stairs(width - 2, height - 2)
                self._dark = dark
                if monsters:
                    self._spawn_creature(KOBOLD, 3, 3)

        return TestEnv(max_turns=max_turns)

    def test_reset_determinism(self) -> None:
        e1, e2 = self._make_env(), self._make_env()
        assert e1.reset(seed=42)[0] == e2.reset(seed=42)[0]

    def test_movement(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        env.step(env.action_spec.index_of("MOVE_E"))
        assert env._player_pos == (2, 1)

    def test_wall_blocks(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        env.step(env.action_spec.index_of("MOVE_N"))  # wall at y=0
        assert env._player_pos == (1, 1)

    def test_reach_goal(self) -> None:
        env = self._make_env(width=5, height=5)
        env.reset(seed=0)
        # (1,1) -> (3,3) via diagonal
        env.step(env.action_spec.index_of("MOVE_SE"))
        _, reward, terminated, _, info = env.step(env.action_spec.index_of("MOVE_SE"))
        assert terminated
        assert reward == 1.0
        assert info.get("goal_reached")

    def test_combat(self) -> None:
        env = self._make_env(monsters=True)
        env.reset(seed=0)
        # Move toward monster at (3,3)
        env.step(env.action_spec.index_of("MOVE_SE"))  # (2,2)
        env.step(env.action_spec.index_of("MOVE_SE"))  # attacks monster at (3,3)
        assert env._player_pos == (2, 2)  # didn't move into monster cell

    def test_dark_room_visibility(self) -> None:
        env = self._make_env(width=9, height=9, dark=True)
        env.reset(seed=0)
        obs = env.get_observation()
        rendered = obs.render()
        # In dark mode, far cells should be spaces
        assert " " in rendered

    def test_system_prompt(self) -> None:
        env = self._make_env()
        prompt = env.system_prompt()
        assert len(prompt) > 50
        assert "MOVE_N" in prompt
