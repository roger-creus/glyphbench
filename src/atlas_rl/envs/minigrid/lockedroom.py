"""MiniGrid LockedRoom and BlockedUnlockPickup environments.

LockedRoom: Three rooms, goal behind locked door, key in another room.
BlockedUnlockPickup: Door blocked by ball, must clear path then unlock.
"""

from __future__ import annotations

from atlas_rl.envs.minigrid.base import DIR_RIGHT, MiniGridBase
from atlas_rl.envs.minigrid.objects import Ball, Box, Door, Goal, Key, Wall


class MiniGridLockedRoomEnv(MiniGridBase):
    """Three rooms in a row. Goal behind locked door, key in another room."""

    def env_id(self) -> str:
        return "atlas_rl/minigrid-lockedroom-v0"

    def _generate_grid(self, seed: int) -> None:
        # 19x7 grid: three rooms of 5-cell interior each
        self._init_grid(19, 7)

        # Vertical walls dividing into 3 rooms
        # Wall at x=6
        for y in range(1, 6):
            self._place_obj(6, y, Wall())
        # Wall at x=12
        for y in range(1, 6):
            self._place_obj(12, y, Wall())

        # Door between left and middle room
        left_door_y = int(self.rng.integers(1, 6))
        self._grid[left_door_y][6] = None
        self._place_obj(6, left_door_y, Door(color="green"))

        # Door between middle and right room
        right_door_y = int(self.rng.integers(1, 6))
        self._grid[right_door_y][12] = None

        # Randomly decide which side has goal (locked) and which has key
        if int(self.rng.integers(0, 2)) == 0:
            # Goal in right room (locked), key in left room
            self._place_obj(12, right_door_y, Door(color="yellow", is_locked=True))
            # Key in left room
            kx = int(self.rng.integers(1, 6))
            ky = int(self.rng.integers(1, 6))
            self._place_obj(kx, ky, Key(color="yellow"))
            # Goal in right room
            gx = int(self.rng.integers(13, 18))
            gy = int(self.rng.integers(1, 6))
            self._place_obj(gx, gy, Goal())
        else:
            # Goal in left room (swap: make left door locked)
            self._grid[left_door_y][6] = None
            self._place_obj(6, left_door_y, Door(color="yellow", is_locked=True))
            self._place_obj(12, right_door_y, Door(color="green"))
            # Key in right room
            kx = int(self.rng.integers(13, 18))
            ky = int(self.rng.integers(1, 6))
            self._place_obj(kx, ky, Key(color="yellow"))
            # Goal in left room
            gx = int(self.rng.integers(1, 6))
            gy = int(self.rng.integers(1, 6))
            while (gx, gy) == (kx, ky):
                gx = int(self.rng.integers(1, 6))
                gy = int(self.rng.integers(1, 6))
            self._place_obj(gx, gy, Goal())

        # Agent in middle room
        ax = int(self.rng.integers(7, 12))
        ay = int(self.rng.integers(1, 6))
        self._place_agent(ax, ay, DIR_RIGHT)

    def _task_description(self) -> str:
        return (
            "Three rooms in a row. The goal (G) is behind a locked yellow door "
            "in one side room. The yellow key (K) is in the other side room "
            "(accessible through an unlocked green door). Navigate to the key, "
            "pick it up, unlock the yellow door, and reach the goal. "
            "Reward = 1 - 0.9 * (steps / max_steps)."
        )


class MiniGridBlockedUnlockPickupEnv(MiniGridBase):
    """Two rooms, door blocked by ball. Clear path, unlock, pick up box."""

    def env_id(self) -> str:
        return "atlas_rl/minigrid-blockedunlockpickup-v0"

    def _generate_grid(self, seed: int) -> None:
        # 11x7 grid: two rooms
        self._init_grid(11, 7)

        # Vertical wall at x=5
        for y in range(1, 6):
            self._place_obj(5, y, Wall())

        # Locked door (not at edges to leave room for ball)
        door_y = int(self.rng.integers(2, 5))
        self._grid[door_y][5] = None
        self._place_obj(5, door_y, Door(color="yellow", is_locked=True))

        # Ball blocking the door (in front of door on left side)
        self._place_obj(4, door_y, Ball(color="blue"))

        # Key in left room (not on ball position)
        while True:
            kx = int(self.rng.integers(1, 5))
            ky = int(self.rng.integers(1, 6))
            if (kx, ky) != (4, door_y):
                break
        self._place_obj(kx, ky, Key(color="yellow"))

        # Agent in left room (not on key or ball)
        while True:
            ax = int(self.rng.integers(1, 5))
            ay = int(self.rng.integers(1, 6))
            if (ax, ay) != (kx, ky) and (ax, ay) != (4, door_y):
                break
        self._place_agent(ax, ay, DIR_RIGHT)

        # Box in right room (target)
        bx = int(self.rng.integers(6, 10))
        by = int(self.rng.integers(1, 6))
        self._place_obj(bx, by, Box(color="green"))

        # Goal in right room (not on box)
        while True:
            gx = int(self.rng.integers(6, 10))
            gy = int(self.rng.integers(1, 6))
            if (gx, gy) != (bx, by):
                break
        self._place_obj(gx, gy, Goal())

    def _task_description(self) -> str:
        return (
            "Two rooms separated by a locked door. A blue ball (O) blocks access "
            "to the door. Move the ball out of the way (PICKUP then DROP elsewhere), "
            "find the yellow key (K), unlock the door (D), then reach the goal (G). "
            "Reward = 1 - 0.9 * (steps / max_steps)."
        )
