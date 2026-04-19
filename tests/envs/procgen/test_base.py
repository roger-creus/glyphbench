"""Tests for ProcgenBase."""

from __future__ import annotations

from atlas_rl.core.action import ActionSpec


class TestProcgenBase:
    def _make_env(
        self, width: int = 40, height: int = 12, max_turns: int = 100, gravity: bool = False
    ):  # type: ignore[return]
        from atlas_rl.envs.procgen.base import ProcgenBase

        class TestGame(ProcgenBase):
            action_spec = ActionSpec(
                names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN"),
                descriptions=(
                    "no-op",
                    "move left",
                    "move right",
                    "move up",
                    "move down",
                ),
            )

            def env_id(self) -> str:
                return "atlas_rl/test-procgen-v0"

            def _generate_level(self, seed: int) -> None:
                self._init_world(width, height)
                self._agent_x, self._agent_y = 2, height - 3
                self._has_gravity = gravity
                # Ground
                for x in range(width):
                    for y in range(height - 2, height):
                        self._set_cell(x, y, "=")
                # Goal
                self._set_cell(width - 3, height - 3, "C")

            def _game_step(self, action_name: str) -> tuple[float, bool, dict]:
                reward, terminated = 0.0, False
                if action_name == "LEFT":
                    self._try_move(-1, 0)
                elif action_name == "RIGHT":
                    self._try_move(1, 0)
                elif action_name == "UP":
                    self._try_move(0, -1)
                elif action_name == "DOWN":
                    self._try_move(0, 1)
                if self._world_at(self._agent_x, self._agent_y) == "C":
                    reward = 10.0
                    terminated = True
                    self._set_cell(self._agent_x, self._agent_y, ".")
                return reward, terminated, {}

        return TestGame(max_turns=max_turns)

    def test_reset_determinism(self) -> None:
        e1, e2 = self._make_env(), self._make_env()
        assert e1.reset(seed=42)[0] == e2.reset(seed=42)[0]

    def test_movement(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        right = env.action_spec.index_of("RIGHT")
        env.step(right)
        assert env._agent_x == 3

    def test_wall_blocks(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        down = env.action_spec.index_of("DOWN")
        env.step(down)  # ground blocks
        assert env._agent_y == env._world_h - 3

    def test_partial_obs_rendering(self) -> None:
        env = self._make_env()
        obs, _ = env.reset(seed=0)
        assert isinstance(obs, str)
        assert "@" in obs

    def test_system_prompt(self) -> None:
        env = self._make_env()
        prompt = env.system_prompt()
        assert "NOOP" in prompt
        assert len(prompt) > 50

    def test_entity_management(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        env._add_entity("enemy", "m", 10, env._world_h - 3, dx=-1)
        assert len(env._entities) == 1
        env.step(env.action_spec.index_of("NOOP"))
        assert env._entities[0].x == 9  # moved left
