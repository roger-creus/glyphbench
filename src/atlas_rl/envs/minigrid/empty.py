"""MiniGrid Empty-5x5 environment.

A 5x5 empty room (7x7 grid including walls). Agent starts at (1,1) facing
right. Goal at (5,5). Agent must navigate to the goal.

Gym ID: atlas_rl/minigrid-empty-5x5-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.ascii_primitives import (
    build_legend,
    draw_box,
    grid_to_string,
    make_empty_grid,
    stamp_sprite,
)
from atlas_rl.core.base_env import BaseAsciiEnv
from atlas_rl.core.observation import GridObservation
from atlas_rl.envs.minigrid.base import (
    DIR_DOWN,
    DIR_LEFT,
    DIR_RIGHT,
    DIR_TO_CHAR,
    DIR_TO_VEC,
    DIR_UP,
    MINIGRID_ACTION_SPEC,
)


class MiniGridEmpty5x5Env(BaseAsciiEnv):
    """MiniGrid Empty-5x5: navigate a 5x5 room to reach the goal.

    Grid is 7x7 (5x5 interior + wall border). Agent starts at grid position
    (1,1) facing right. Goal is at grid position (5,5).

    Actions: TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, PICKUP, DROP, TOGGLE, DONE
    Only the first three are meaningful; the rest are no-ops.

    Reward: 1 - 0.9 * (step_count / max_steps) on reaching goal, 0 otherwise.
    """

    action_spec = MINIGRID_ACTION_SPEC
    noop_action_name = "DONE"

    _GRID_W = 7
    _GRID_H = 7
    _GOAL_X = 5  # grid coords
    _GOAL_Y = 5  # grid coords
    _START_X = 1  # grid coords
    _START_Y = 1  # grid coords

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)
        self._agent_x: int = self._START_X
        self._agent_y: int = self._START_Y
        self._agent_dir: int = DIR_RIGHT

    def env_id(self) -> str:
        return "atlas_rl/minigrid-empty-5x5-v0"

    def system_prompt(self) -> str:
        return (
            "You are playing MiniGrid Empty-5x5.\n\n"
            "TASK\n"
            "Navigate a 5x5 room from the top-left corner to the goal G in the "
            "bottom-right corner. You earn a reward based on how quickly you reach "
            "the goal: reward = 1 - 0.9 * (steps_taken / 100). The faster you "
            "reach the goal, the higher your reward.\n\n"
            "GRID\n"
            "The room is a 7x7 grid (5x5 interior + walls). You are one of >, v, <, ^ "
            "showing your facing direction. G is the goal. . is floor. Walls are "
            "shown as +, -, |.\n\n"
            "MOVEMENT\n"
            "You move relative to your facing direction. MOVE_FORWARD advances one "
            "cell in the direction you face. TURN_LEFT and TURN_RIGHT rotate you 90 "
            "degrees. Bumping into a wall does nothing.\n\n"
            "ACTIONS: PICKUP, DROP, TOGGLE, DONE are no-ops in this environment.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        self._agent_x = self._START_X
        self._agent_y = self._START_Y
        self._agent_dir = DIR_RIGHT
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]

        if name == "TURN_LEFT":
            self._agent_dir = (self._agent_dir - 1) % 4
        elif name == "TURN_RIGHT":
            self._agent_dir = (self._agent_dir + 1) % 4
        elif name == "MOVE_FORWARD":
            dx, dy = DIR_TO_VEC[self._agent_dir]
            new_x = self._agent_x + dx
            new_y = self._agent_y + dy
            # Bounds check: interior is cols 1-5, rows 1-5
            if 1 <= new_x <= 5 and 1 <= new_y <= 5:
                self._agent_x = new_x
                self._agent_y = new_y
        # PICKUP, DROP, TOGGLE, DONE: no-ops

        reached_goal = (
            self._agent_x == self._GOAL_X and self._agent_y == self._GOAL_Y
        )
        # step_count is self._turn + 1 because BaseAsciiEnv increments _turn
        # AFTER _step returns. So at the time _step is called, _turn is the
        # 0-based index of this step. The reward formula uses the 1-based step
        # count, which is _turn + 1.
        step_count = self._turn + 1
        reward = (1.0 - 0.9 * (step_count / self.max_turns)) if reached_goal else 0.0
        terminated = reached_goal

        info: dict[str, Any] = {
            "agent_pos": (self._agent_x - 1, self._agent_y - 1),  # interior coords
            "goal_reached": reached_goal,
            "steps_to_goal": step_count if reached_goal else -1,
        }

        return self._render_current_observation(), reward, terminated, False, info

    def _render_current_observation(self) -> GridObservation:
        grid = make_empty_grid(self._GRID_W, self._GRID_H, fill=".")
        draw_box(grid, 0, 0, self._GRID_W - 1, self._GRID_H - 1)
        stamp_sprite(grid, self._GOAL_X, self._GOAL_Y, "G")
        stamp_sprite(grid, self._agent_x, self._agent_y, DIR_TO_CHAR[self._agent_dir])

        facing_name = {DIR_RIGHT: "RIGHT", DIR_DOWN: "DOWN", DIR_LEFT: "LEFT", DIR_UP: "UP"}
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Facing: {facing_name[self._agent_dir]}    "
            f"Position: ({self._agent_x - 1}, {self._agent_y - 1})"
        )

        legend = build_legend({
            ">": "you, facing right",
            "v": "you, facing down",
            "<": "you, facing left",
            "^": "you, facing up",
            "G": "goal",
            ".": "floor",
            "+": "wall corner",
            "-": "wall (horizontal)",
            "|": "wall (vertical)",
        })

        return GridObservation(grid=grid_to_string(grid), legend=legend, hud=hud, message="")
