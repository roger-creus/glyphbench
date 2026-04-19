"""MiniGrid Empty-5x5 environment.

A 5x5 empty room (7x7 grid including walls). Agent starts at (1,1) facing
right. Goal at (5,5). Agent must navigate to the goal.

Gym ID: atlas_rl/minigrid-empty-5x5-v0
"""

from __future__ import annotations

from atlas_rl.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from atlas_rl.envs.minigrid.objects import Goal


class MiniGridEmpty5x5Env(MiniGridBase):
    """MiniGrid Empty-5x5: navigate a 5x5 room to reach the goal.

    Grid is 7x7 (5x5 interior + wall border). Agent starts at grid position
    (1,1) facing right. Goal is at grid position (5,5).

    Actions: TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, PICKUP, DROP, TOGGLE, DONE
    Only the first three are meaningful; the rest are no-ops.

    Reward: 1 - 0.9 * (step_count / max_steps) on reaching goal, 0 otherwise.
    """

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "atlas_rl/minigrid-empty-5x5-v0"

    def _generate_grid(self, seed: int) -> None:
        self._init_grid(7, 7)
        self._place_agent(1, 1, DIR_RIGHT)
        self._place_obj(5, 5, Goal())

    def _task_description(self) -> str:
        return (
            "Navigate a 5x5 room from the top-left corner to the goal G in the "
            "bottom-right corner. You earn a reward based on how quickly you reach "
            "the goal: reward = 1 - 0.9 * (steps_taken / 100). The faster you "
            "reach the goal, the higher your reward."
        )
