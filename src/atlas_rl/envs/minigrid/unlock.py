"""MiniGrid Unlock and UnlockPickup environments.

Unlock: Pick up key, unlock door, reach goal.
UnlockPickup: Pick up key, unlock door, pick up box behind door.
"""

from __future__ import annotations

from atlas_rl.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from atlas_rl.envs.minigrid.objects import Box, Door, Goal, Key, Wall


class MiniGridUnlockEnv(MiniGridBase):
    """Two rooms separated by a locked door. Pick up key, unlock, reach goal."""

    def env_id(self) -> str:
        return "atlas_rl/minigrid-unlock-v0"

    def _generate_grid(self, seed: int) -> None:
        # 11x7 grid: two 4x5-interior rooms separated by a wall with locked door
        self._init_grid(11, 7)

        # Vertical dividing wall at x=5
        for y in range(1, 6):
            self._place_obj(5, y, Wall())

        # Locked door in the dividing wall
        door_y = int(self.rng.integers(1, 6))
        self._grid[door_y][5] = None
        self._place_obj(5, door_y, Door(color="yellow", is_locked=True))

        # Key in left room
        key_x = int(self.rng.integers(1, 5))
        key_y = int(self.rng.integers(1, 6))
        self._place_obj(key_x, key_y, Key(color="yellow"))

        # Agent in left room, not on key
        while True:
            ax = int(self.rng.integers(1, 5))
            ay = int(self.rng.integers(1, 6))
            if (ax, ay) != (key_x, key_y):
                break
        self._place_agent(ax, ay, DIR_RIGHT)

        # Goal in right room
        gx = int(self.rng.integers(6, 10))
        gy = int(self.rng.integers(1, 6))
        self._place_obj(gx, gy, Goal())

    def _task_description(self) -> str:
        return (
            "Two rooms separated by a locked door. Find the yellow key (K), "
            "pick it up with PICKUP, face the locked door (D) and TOGGLE to "
            "unlock it, then navigate to the goal (G). "
            "Reward = 1 - 0.9 * (steps / max_steps)."
        )


class MiniGridUnlockPickupEnv(MiniGridBase):
    """Two rooms separated by a locked door. Unlock, then pick up a box."""

    def env_id(self) -> str:
        return "atlas_rl/minigrid-unlockpickup-v0"

    def _generate_grid(self, seed: int) -> None:
        self._init_grid(11, 7)

        # Vertical dividing wall at x=5
        for y in range(1, 6):
            self._place_obj(5, y, Wall())

        # Locked door in the dividing wall
        door_y = int(self.rng.integers(1, 6))
        self._grid[door_y][5] = None
        self._place_obj(5, door_y, Door(color="yellow", is_locked=True))

        # Key in left room
        key_x = int(self.rng.integers(1, 5))
        key_y = int(self.rng.integers(1, 6))
        self._place_obj(key_x, key_y, Key(color="yellow"))

        # Agent in left room, not on key
        while True:
            ax = int(self.rng.integers(1, 5))
            ay = int(self.rng.integers(1, 6))
            if (ax, ay) != (key_x, key_y):
                break
        self._place_agent(ax, ay, DIR_RIGHT)

        # Box in right room (target to pick up)
        bx = int(self.rng.integers(6, 10))
        by = int(self.rng.integers(1, 6))
        self._place_obj(bx, by, Box(color="green"))

        # Goal in right room (for termination), not on box
        while True:
            gx = int(self.rng.integers(6, 10))
            gy = int(self.rng.integers(1, 6))
            if (gx, gy) != (bx, by):
                break
        self._place_obj(gx, gy, Goal())

    def _task_description(self) -> str:
        return (
            "Two rooms separated by a locked door. Find the yellow key (K), "
            "unlock the door (D), then pick up the green box (B) in the other "
            "room and reach the goal (G). "
            "Reward = 1 - 0.9 * (steps / max_steps)."
        )
