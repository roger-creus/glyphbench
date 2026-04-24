"""Tests for MiniHackBase shared functionality."""

from __future__ import annotations

from glyphbench.core import make_env
import glyphbench.envs.minihack  # register envs

from glyphbench.envs.minihack.creatures import KOBOLD


class TestMiniHackBase:
    def _make_env(
        self,
        width: int = 7,
        height: int = 7,
        max_turns: int = 100,
        dark: bool = False,
        monsters: bool = False,
    ):  # type: ignore[no-untyped-def]
        from glyphbench.envs.minihack.base import MiniHackBase

        class TestEnv(MiniHackBase):
            def env_id(self) -> str:
                return "glyphbench/test-minihack-v0"

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
        assert e1.reset(42)[0] == e2.reset(42)[0]

    def test_movement(self) -> None:
        env = self._make_env()
        env.reset(0)
        env.step(env.action_spec.index_of("MOVE_E"))
        assert env._player_pos == (2, 1)

    def test_wall_blocks(self) -> None:
        env = self._make_env()
        env.reset(0)
        env.step(env.action_spec.index_of("MOVE_N"))  # wall at y=0
        assert env._player_pos == (1, 1)

    def test_reach_goal(self) -> None:
        env = self._make_env(width=5, height=5)
        env.reset(0)
        # (1,1) -> (3,3) via diagonal
        env.step(env.action_spec.index_of("MOVE_SE"))
        _, reward, terminated, _, info = env.step(env.action_spec.index_of("MOVE_SE"))
        assert terminated
        assert reward == 1.0
        assert info.get("goal_reached")

    def test_combat(self) -> None:
        env = self._make_env(monsters=True)
        env.reset(0)
        # Move toward monster at (3,3)
        env.step(env.action_spec.index_of("MOVE_SE"))  # (2,2)
        env.step(env.action_spec.index_of("MOVE_SE"))  # attacks monster at (3,3)
        assert env._player_pos == (2, 2)  # didn't move into monster cell

    def test_dark_room_visibility(self) -> None:
        env = self._make_env(width=9, height=9, dark=True)
        env.reset(0)
        obs = env.get_observation()
        rendered = obs.render()
        # In dark mode, far cells should be spaces
        assert " " in rendered

    def test_system_prompt(self) -> None:
        env = self._make_env()
        prompt = env.system_prompt()
        assert len(prompt) > 50
        assert "MOVE_N" in prompt

    def test_pickup_item(self) -> None:
        from glyphbench.envs.minihack.base import MiniHackBase
        from glyphbench.envs.minihack.items import FOOD_RATION

        class ItemEnv(MiniHackBase):
            def env_id(self) -> str:
                return "glyphbench/test-items-v0"

            def _generate_level(self, seed: int) -> None:
                self._init_grid(7, 7)
                self._place_player(1, 1)
                self._place_stairs(5, 5)
                self._place_item(1, 1, FOOD_RATION)

        env = ItemEnv(max_turns=100)
        env.reset(0)
        pickup = env.action_spec.index_of("PICKUP")
        env.step(pickup)
        assert len(env._inventory) == 1
        assert env._inventory[0].name == "food ration"

    def test_eat_food(self) -> None:
        from glyphbench.envs.minihack.base import MiniHackBase
        from glyphbench.envs.minihack.items import FOOD_RATION

        class EatEnv(MiniHackBase):
            def env_id(self) -> str:
                return "glyphbench/test-eat-v0"

            def _generate_level(self, seed: int) -> None:
                self._init_grid(7, 7)
                self._place_player(1, 1)
                self._place_stairs(5, 5)
                self._place_item(1, 1, FOOD_RATION)

        env = EatEnv(max_turns=100)
        env.reset(0)
        env.step(env.action_spec.index_of("PICKUP"))
        env.step(env.action_spec.index_of("EAT"))
        assert len(env._inventory) == 0  # food consumed

    def test_drop_item(self) -> None:
        from glyphbench.envs.minihack.base import MiniHackBase
        from glyphbench.envs.minihack.items import SWORD

        class DropEnv(MiniHackBase):
            def env_id(self) -> str:
                return "glyphbench/test-drop-v0"

            def _generate_level(self, seed: int) -> None:
                self._init_grid(7, 7)
                self._place_player(1, 1)
                self._place_stairs(5, 5)
                self._place_item(1, 1, SWORD)

        env = DropEnv(max_turns=100)
        env.reset(0)
        env.step(env.action_spec.index_of("PICKUP"))
        assert len(env._inventory) == 1
        env.step(env.action_spec.index_of("DROP"))
        assert len(env._inventory) == 0
        assert (1, 1) in env._floor_items

    def test_wield_weapon(self) -> None:
        from glyphbench.envs.minihack.base import MiniHackBase
        from glyphbench.envs.minihack.items import SWORD

        class WieldEnv(MiniHackBase):
            def env_id(self) -> str:
                return "glyphbench/test-wield-v0"

            def _generate_level(self, seed: int) -> None:
                self._init_grid(7, 7)
                self._place_player(1, 1)
                self._place_stairs(5, 5)
                self._place_item(1, 1, SWORD)

        env = WieldEnv(max_turns=100)
        env.reset(0)
        env.step(env.action_spec.index_of("PICKUP"))
        env.step(env.action_spec.index_of("WIELD"))
        assert env._wielding is not None
        assert env._wielding.name == "long sword"
        # Wielding does NOT remove from inventory
        assert len(env._inventory) == 1

    def test_read_scroll(self) -> None:
        from glyphbench.envs.minihack.base import MiniHackBase
        from glyphbench.envs.minihack.items import SCROLL_LIGHT

        class ReadEnv(MiniHackBase):
            def env_id(self) -> str:
                return "glyphbench/test-read-v0"

            def _generate_level(self, seed: int) -> None:
                self._init_grid(7, 7)
                self._place_player(1, 1)
                self._place_stairs(5, 5)
                self._place_item(1, 1, SCROLL_LIGHT)

        env = ReadEnv(max_turns=100)
        env.reset(0)
        env.step(env.action_spec.index_of("PICKUP"))
        env.step(env.action_spec.index_of("READ"))
        assert len(env._inventory) == 0  # scroll consumed

    def test_quaff_potion(self) -> None:
        from glyphbench.envs.minihack.base import MiniHackBase
        from glyphbench.envs.minihack.items import POTION_HEALING

        class QuaffEnv(MiniHackBase):
            def env_id(self) -> str:
                return "glyphbench/test-quaff-v0"

            def _generate_level(self, seed: int) -> None:
                self._init_grid(7, 7)
                self._place_player(1, 1)
                self._place_stairs(5, 5)
                self._place_item(1, 1, POTION_HEALING)

        env = QuaffEnv(max_turns=100)
        env.reset(0)
        env.step(env.action_spec.index_of("PICKUP"))
        env.step(env.action_spec.index_of("QUAFF"))
        assert len(env._inventory) == 0  # potion consumed

    def test_zap_wand(self) -> None:
        from glyphbench.envs.minihack.base import MiniHackBase
        from glyphbench.envs.minihack.items import WAND_DEATH

        class ZapEnv(MiniHackBase):
            def env_id(self) -> str:
                return "glyphbench/test-zap-v0"

            def _generate_level(self, seed: int) -> None:
                self._init_grid(7, 7)
                self._place_player(1, 1)
                self._place_stairs(5, 5)
                self._place_item(1, 1, WAND_DEATH)

        env = ZapEnv(max_turns=100)
        env.reset(0)
        env.step(env.action_spec.index_of("PICKUP"))
        env.step(env.action_spec.index_of("ZAP"))
        assert len(env._inventory) == 0  # wand consumed

    def test_pray_action(self) -> None:
        env = self._make_env()
        env.reset(0)
        env.step(env.action_spec.index_of("PRAY"))
        obs = env.get_observation()
        assert "pray" in obs.message.lower()

    def test_floor_item_rendered(self) -> None:
        from glyphbench.envs.minihack.base import MiniHackBase
        from glyphbench.envs.minihack.items import FOOD_RATION

        class RenderEnv(MiniHackBase):
            def env_id(self) -> str:
                return "glyphbench/test-render-item-v0"

            def _generate_level(self, seed: int) -> None:
                self._init_grid(7, 7)
                self._place_player(1, 1)
                self._place_stairs(5, 5)
                self._place_item(3, 3, FOOD_RATION)

        env = RenderEnv(max_turns=100)
        env.reset(0)
        obs = env.get_observation()
        assert "%" in obs.grid

    def test_inventory_clears_on_reset(self) -> None:
        from glyphbench.envs.minihack.base import MiniHackBase
        from glyphbench.envs.minihack.items import FOOD_RATION

        class ResetEnv(MiniHackBase):
            def env_id(self) -> str:
                return "glyphbench/test-reset-inv-v0"

            def _generate_level(self, seed: int) -> None:
                self._init_grid(7, 7)
                self._place_player(1, 1)
                self._place_stairs(5, 5)
                self._place_item(1, 1, FOOD_RATION)

        env = ResetEnv(max_turns=100)
        env.reset(0)
        env.step(env.action_spec.index_of("PICKUP"))
        assert len(env._inventory) == 1
        env.reset(0)
        assert len(env._inventory) == 0
