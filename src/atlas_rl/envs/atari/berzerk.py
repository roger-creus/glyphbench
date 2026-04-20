"""Atari Berzerk environment.

Navigate rooms, shoot robots, avoid walls. Exit through doors.

Gym ID: atlas_rl/atari-berzerk-v0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas_rl.core.action import ActionSpec
from atlas_rl.core.observation import GridObservation

from .base import AtariBase, AtariEntity

_DIRS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

_W = 20
_H = 16
_MAX_ROBOTS = 8
_BULLET_CHAR = "!"
_ROBOT_CHAR = "r"
_EXIT_CHAR = "D"


class BerzerkEnv(AtariBase):
    """Berzerk: shoot robots, escape rooms, don't touch walls.

    Actions: NOOP, FIRE, UP, RIGHT, LEFT, DOWN,
             UP_FIRE, DOWN_FIRE, LEFT_FIRE, RIGHT_FIRE
    Reward: +50 per robot destroyed.
    Level clears when all robots dead or agent exits through door.
    """

    action_spec = ActionSpec(
        names=(
            "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
            "UP_FIRE", "DOWN_FIRE", "LEFT_FIRE", "RIGHT_FIRE",
        ),
        descriptions=(
            "do nothing this step",
            "fire bullet in last-faced direction",
            "move up one cell",
            "move right one cell",
            "move left one cell",
            "move down one cell",
            "move up and fire",
            "move down and fire",
            "move left and fire",
            "move right and fire",
        ),
    )

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._facing: tuple[int, int] = (1, 0)
        self._robots_killed: int = 0
        self._total_robots: int = 0

    def env_id(self) -> str:
        return "atlas_rl/atari-berzerk-v0"

    def _task_description(self) -> str:
        return (
            "Shoot robots (r) and exit through the door (D). "
            "Touching a wall (#) or a robot kills you. "
            "Fire bullets (!) to destroy robots."
        )

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "#": "wall",
            " ": "empty",
            "r": "robot enemy",
            "!": "bullet",
            "D": "exit door",
        }.get(ch, ch)

    def _generate_level(self, seed: int) -> None:
        rng = np.random.default_rng(seed + self._level * 1000)
        self._init_grid(_W, _H)
        self._entities = []
        self._robots_killed = 0
        self._facing = (1, 0)

        # Border walls
        for x in range(_W):
            self._set_cell(x, 0, "#")
            self._set_cell(x, _H - 1, "#")
        for y in range(_H):
            self._set_cell(0, y, "#")
            self._set_cell(_W - 1, y, "#")

        # Interior walls: random vertical and horizontal segments
        n_walls = int(rng.integers(3, 7))
        for _ in range(n_walls):
            if rng.random() < 0.5:
                # Vertical wall
                wx = int(rng.integers(3, _W - 3))
                wy = int(rng.integers(2, _H - 4))
                length = int(rng.integers(2, 5))
                for j in range(length):
                    if wy + j < _H - 1:
                        self._set_cell(wx, wy + j, "#")
            else:
                # Horizontal wall
                wx = int(rng.integers(3, _W - 5))
                wy = int(rng.integers(2, _H - 3))
                length = int(rng.integers(2, 5))
                for j in range(length):
                    if wx + j < _W - 1:
                        self._set_cell(wx + j, wy, "#")

        # Place exit door on right wall
        door_y = int(rng.integers(2, _H - 2))
        self._set_cell(_W - 1, door_y, _EXIT_CHAR)

        # Place player in bottom-left area
        self._player_x = 2
        self._player_y = _H - 3
        # Make sure player start is clear
        self._set_cell(self._player_x, self._player_y, " ")

        # Place robots
        n_robots = min(_MAX_ROBOTS, 3 + self._level)
        self._total_robots = 0
        for _ in range(n_robots):
            for _attempt in range(20):
                rx = int(rng.integers(3, _W - 3))
                ry = int(rng.integers(2, _H - 3))
                if (
                    self._grid_at(rx, ry) == " "
                    and abs(rx - self._player_x) + abs(ry - self._player_y) > 4
                ):
                    robot = self._add_entity("robot", _ROBOT_CHAR, rx, ry)
                    robot.data["shoot_timer"] = int(rng.integers(5, 15))
                    self._total_robots += 1
                    break

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Parse action
        move_dir: tuple[int, int] | None = None
        fire = False
        if action_name == "NOOP":
            pass
        elif action_name == "FIRE":
            fire = True
        elif action_name in _DIRS:
            move_dir = _DIRS[action_name]
            self._facing = move_dir
            self._player_dir = move_dir
        elif action_name == "UP_FIRE":
            move_dir = _DIRS["UP"]
            self._facing = move_dir
            self._player_dir = move_dir
            fire = True
        elif action_name == "DOWN_FIRE":
            move_dir = _DIRS["DOWN"]
            self._facing = move_dir
            self._player_dir = move_dir
            fire = True
        elif action_name == "LEFT_FIRE":
            move_dir = _DIRS["LEFT"]
            self._facing = move_dir
            self._player_dir = move_dir
            fire = True
        elif action_name == "RIGHT_FIRE":
            move_dir = _DIRS["RIGHT"]
            self._facing = move_dir
            self._player_dir = move_dir
            fire = True

        # Move player
        if move_dir is not None:
            nx = self._player_x + move_dir[0]
            ny = self._player_y + move_dir[1]
            cell = self._grid_at(nx, ny)
            if cell == _EXIT_CHAR:
                # Exit room - level complete
                self._player_x, self._player_y = nx, ny
                self._level += 1
                self._message = "Room cleared!"
                self._generate_level(self._level * 7919)
                info["room_cleared"] = True
                return reward, False, info
            if not self._is_solid(nx, ny):
                self._player_x, self._player_y = nx, ny

        # Check if player touched a wall (electrocution)
        if self._is_solid(self._player_x, self._player_y):
            self._on_life_lost()
            reward -= 50.0
            if not self._game_over:
                self._player_x = 2
                self._player_y = _H - 3
            return reward, self._game_over, info

        # Fire bullet
        if fire:
            bx = self._player_x + self._facing[0]
            by = self._player_y + self._facing[1]
            if not self._is_solid(bx, by):
                b = self._add_entity("bullet", _BULLET_CHAR, bx, by)
                b.dx = self._facing[0]
                b.dy = self._facing[1]
                b.data["owner"] = "player"

        # Move bullets
        new_bullets: list[AtariEntity] = []
        for e in self._entities:
            if e.etype != "bullet" or not e.alive:
                continue
            e.x += e.dx
            e.y += e.dy
            if self._is_solid(e.x, e.y) or e.x <= 0 or e.x >= _W - 1 or e.y <= 0 or e.y >= _H - 1:
                e.alive = False
            else:
                new_bullets.append(e)

        # Robot AI
        for e in self._entities:
            if e.etype != "robot" or not e.alive:
                continue
            # Simple movement toward player
            if self.rng.random() < 0.3:
                dx = 0
                dy = 0
                if self._player_x > e.x:
                    dx = 1
                elif self._player_x < e.x:
                    dx = -1
                if self._player_y > e.y:
                    dy = 1
                elif self._player_y < e.y:
                    dy = -1
                # Pick one axis
                if self.rng.random() < 0.5:
                    dy = 0
                else:
                    dx = 0
                nx, ny = e.x + dx, e.y + dy
                if not self._is_solid(nx, ny):
                    e.x, e.y = nx, ny

            # Robot shooting
            e.data["shoot_timer"] = e.data.get("shoot_timer", 10) - 1
            if e.data["shoot_timer"] <= 0:
                e.data["shoot_timer"] = int(self.rng.integers(8, 20))
                # Fire toward player
                ddx = 0
                ddy = 0
                if self._player_x > e.x:
                    ddx = 1
                elif self._player_x < e.x:
                    ddx = -1
                if self._player_y > e.y:
                    ddy = 1
                elif self._player_y < e.y:
                    ddy = -1
                if ddx != 0 or ddy != 0:
                    # Pick one axis for bullet direction
                    if abs(self._player_x - e.x) > abs(self._player_y - e.y):
                        ddy = 0
                    else:
                        ddx = 0
                    bx, by = e.x + ddx, e.y + ddy
                    if not self._is_solid(bx, by):
                        rb = self._add_entity("bullet", _BULLET_CHAR, bx, by)
                        rb.dx = ddx
                        rb.dy = ddy
                        rb.data["owner"] = "robot"

        # Check bullet-robot collisions
        for b in self._entities:
            if b.etype != "bullet" or not b.alive:
                continue
            if b.data.get("owner") == "robot":
                # Robot bullets hit player
                if b.x == self._player_x and b.y == self._player_y:
                    b.alive = False
                    self._on_life_lost()
                    reward -= 50.0
                    if not self._game_over:
                        self._player_x = 2
                        self._player_y = _H - 3
                continue
            # Player bullets hit robots
            for r in self._entities:
                if r.etype != "robot" or not r.alive:
                    continue
                if b.x == r.x and b.y == r.y:
                    b.alive = False
                    r.alive = False
                    self._robots_killed += 1
                    self._on_point_scored(50)
                    reward += 50.0

        # Check robot-player collision
        for e in self._entities:
            if e.etype == "robot" and e.alive and e.x == self._player_x and e.y == self._player_y:
                    self._on_life_lost()
                    reward -= 50.0
                    if not self._game_over:
                        self._player_x = 2
                        self._player_y = _H - 3
                    break

        # Clean dead entities
        self._entities = [e for e in self._entities if e.alive]

        # Check if all robots are dead
        robots_alive = sum(1 for e in self._entities if e.etype == "robot" and e.alive)
        if robots_alive == 0 and self._total_robots > 0:
            self._message = "All robots destroyed!"
            self._level += 1
            self._generate_level(self._level * 7919)
            info["room_cleared"] = True

        info["robots_alive"] = robots_alive
        info["robots_killed"] = self._robots_killed
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        dname = self._DIR_NAMES.get(self._facing, "none")
        robots = sum(
            1 for e in self._entities
            if e.etype == "robot" and e.alive
        )
        extra = (
            f"Facing: {dname}  Robots: {robots}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _advance_entities(self) -> None:
        """Override: entities are moved in _game_step."""
        pass
