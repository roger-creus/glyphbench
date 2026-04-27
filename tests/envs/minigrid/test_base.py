"""Tests for MiniGridBase shared functionality."""

from __future__ import annotations

from glyphbench.envs.minigrid.objects import Door, Goal, Key


class TestMiniGridBase:
    def _make_env(self, width: int = 7, height: int = 7, max_turns: int = 100):
        from glyphbench.envs.minigrid.base import MiniGridBase

        class TestEnv(MiniGridBase):
            def env_id(self) -> str:
                return "glyphbench/test-minigrid-v0"

            def _generate_grid(self, seed: int) -> None:
                self._init_grid(width, height)
                self._place_agent(1, 1, direction=0)
                self._place_obj(width - 2, height - 2, Goal())

        return TestEnv(max_turns=max_turns)

    def test_reset_determinism(self) -> None:
        e1 = self._make_env()
        e2 = self._make_env()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    def test_step_determinism(self) -> None:
        e1 = self._make_env()
        e2 = self._make_env()
        e1.reset(seed=0)
        e2.reset(seed=0)
        fwd = e1.action_spec.index_of("MOVE_FORWARD")
        for _ in range(3):
            o1, r1, t1, tr1, _ = e1.step(fwd)
            o2, r2, t2, tr2, _ = e2.step(fwd)
            assert o1 == o2
            assert r1 == r2

    def test_turn_left_right(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        left = env.action_spec.index_of("TURN_LEFT")
        right = env.action_spec.index_of("TURN_RIGHT")
        env.step(left)
        assert env._agent_dir == 3  # RIGHT(0) -> UP(3)
        env.step(right)
        assert env._agent_dir == 0  # UP(3) -> RIGHT(0)

    def test_move_forward(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        fwd = env.action_spec.index_of("MOVE_FORWARD")
        env.step(fwd)  # facing right, (1,1) -> (2,1)
        assert env._agent_pos == (2, 1)

    def test_wall_blocks_movement(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        left = env.action_spec.index_of("TURN_LEFT")
        fwd = env.action_spec.index_of("MOVE_FORWARD")
        env.step(left)  # face up
        env.step(fwd)   # wall at y=0
        assert env._agent_pos == (1, 1)

    def test_action_spec(self) -> None:
        env = self._make_env()
        assert env.action_spec.n == 7
        assert "TURN_LEFT" in env.action_spec.names
        assert "MOVE_FORWARD" in env.action_spec.names
        assert "PICKUP" in env.action_spec.names

    def test_noop_action(self) -> None:
        env = self._make_env()
        assert env.noop_action_name == "DONE"

    def test_grid_rendering_contains_agent_and_goal(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        obs = env.get_observation()
        rendered = obs.render()
        assert "→" in rendered
        assert "★" in rendered

    def test_hud_contains_step_and_carrying(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        obs = env.get_observation()
        assert "Step" in obs.hud
        assert "Carrying" in obs.hud

    def test_system_prompt_nonempty(self) -> None:
        env = self._make_env()
        prompt = env.system_prompt()
        assert len(prompt) > 100
        assert "MOVE_FORWARD" in prompt

    def test_reach_goal_terminates(self) -> None:
        env = self._make_env(width=5, height=5, max_turns=100)
        env.reset(seed=0)
        fwd = env.action_spec.index_of("MOVE_FORWARD")
        right = env.action_spec.index_of("TURN_RIGHT")
        # Agent at (1,1) facing right, goal at (3,3)
        # Move right to (3,1)
        env.step(fwd)
        env.step(fwd)
        # Turn down
        env.step(right)
        # Move down to (3,3)
        env.step(fwd)
        _, reward, terminated, _, info = env.step(fwd)
        assert terminated
        assert reward > 0
        assert info.get("goal_reached") is True

    def test_pickup_and_drop(self) -> None:
        from glyphbench.envs.minigrid.base import MiniGridBase

        class PickupEnv(MiniGridBase):
            def env_id(self) -> str:
                return "glyphbench/test-pickup-v0"

            def _generate_grid(self, seed: int) -> None:
                self._init_grid(7, 7)
                self._place_agent(1, 1, direction=0)  # facing right
                self._place_obj(2, 1, Key(color="red"))  # key ahead
                self._place_obj(5, 5, Goal())

        env = PickupEnv(max_turns=100)
        env.reset(seed=0)
        pickup = env.action_spec.index_of("PICKUP")
        drop = env.action_spec.index_of("DROP")

        # Pick up key
        env.step(pickup)
        assert env._carrying is not None
        assert env._carrying.obj_type == "key"
        assert env._get_obj(2, 1) is None  # removed from grid

        # Drop key
        env.step(drop)
        assert env._carrying is None
        assert env._get_obj(2, 1) is not None  # back on grid

    def test_toggle_door(self) -> None:
        from glyphbench.envs.minigrid.base import MiniGridBase

        class DoorEnv(MiniGridBase):
            def env_id(self) -> str:
                return "glyphbench/test-door-v0"

            def _generate_grid(self, seed: int) -> None:
                self._init_grid(7, 7)
                self._place_agent(1, 1, direction=0)
                self._place_obj(2, 1, Door(color="red"))
                self._place_obj(5, 5, Goal())

        env = DoorEnv(max_turns=100)
        env.reset(seed=0)
        toggle = env.action_spec.index_of("TOGGLE")
        fwd = env.action_spec.index_of("MOVE_FORWARD")

        # Door ahead is closed, can't walk through
        env.step(fwd)
        assert env._agent_pos == (1, 1)  # blocked

        # Toggle to open
        env.step(toggle)
        door = env._get_obj(2, 1)
        assert isinstance(door, Door)
        assert door.is_open is True

        # Now can walk through
        env.step(fwd)
        assert env._agent_pos == (2, 1)
