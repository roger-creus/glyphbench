"""Procgen Chaser (Pac-Man) environment.

Procedural maze with pellets to collect, enemies that chase, and a power
pellet that temporarily makes enemies flee.

Gym ID: atlas_rl/procgen-chaser-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec
from atlas_rl.core.observation import GridObservation
from atlas_rl.envs.procgen.base import Entity, ProcgenBase

_PELLET = "."
_POWER = "O"
_FLOOR = " "


class ChaserEnv(ProcgenBase):
    """Procgen Chaser: Pac-Man-style pellet collection with enemies."""

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "UP", "DOWN"),
        descriptions=(
            "do nothing",
            "move left",
            "move right",
            "move up",
            "move down",
        ),
    )

    MAZE_W = 21
    MAZE_H = 21

    def env_id(self) -> str:
        return "atlas_rl/procgen-chaser-v0"

    def _generate_level(self, seed: int) -> None:
        w, h = self.MAZE_W, self.MAZE_H
        self._init_world(w, h, fill="#")
        self._gen_maze(w, h)

        # Place pellets on all floor cells
        self._pellet_count = 0
        for yy in range(h):
            for xx in range(w):
                if self._world_at(xx, yy) == _FLOOR:
                    self._set_cell(xx, yy, _PELLET)
                    self._pellet_count += 1

        # Agent at (1, 1)
        self._agent_x, self._agent_y = 1, 1
        self._set_cell(1, 1, _FLOOR)  # remove pellet at start
        self._pellet_count -= 1

        # Place 1 power pellet in a random floor cell far from agent
        floor_cells: list[tuple[int, int]] = []
        for yy in range(1, h - 1):
            for xx in range(1, w - 1):
                if self._world_at(xx, yy) == _PELLET:
                    dist = abs(xx - 1) + abs(yy - 1)
                    if dist > 6:
                        floor_cells.append((xx, yy))
        if floor_cells:
            idx = int(self.rng.integers(0, len(floor_cells)))
            px, py = floor_cells[idx]
            self._set_cell(px, py, _POWER)
            self._pellet_count -= 1  # power pellet replaces a regular pellet

        # Place 3 enemies in far corners / random positions
        enemy_positions = [
            (w - 2, 1),
            (1, h - 2),
            (w - 2, h - 2),
        ]
        for ex, ey in enemy_positions:
            if self._world_at(ex, ey) in (_PELLET, _FLOOR, _POWER):
                if self._world_at(ex, ey) == _PELLET:
                    self._pellet_count -= 1
                elif self._world_at(ex, ey) == _POWER:
                    pass  # power pellet stays logically
                self._set_cell(ex, ey, _FLOOR)
                self._add_entity(
                    "ghost", "E", ex, ey,
                    data={"scared": False, "home_x": ex, "home_y": ey},
                )

        self._power_timer = 0
        self._enemies_eaten = 0

    def _gen_maze(self, w: int, h: int) -> None:
        """Carve a maze with wider corridors for gameplay."""
        # Start with walls, carve passages
        self._set_cell(1, 1, _FLOOR)
        stack = [(1, 1)]
        while stack:
            cx, cy = stack[-1]
            neighbors: list[tuple[int, int, int, int]] = []
            for ddx, ddy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + ddx, cy + ddy
                if (
                    1 <= nx < w - 1
                    and 1 <= ny < h - 1
                    and self._world_at(nx, ny) == "#"
                ):
                    neighbors.append((nx, ny, cx + ddx // 2, cy + ddy // 2))
            if neighbors:
                idx = int(self.rng.integers(0, len(neighbors)))
                nx, ny, wx, wy = neighbors[idx]
                self._set_cell(wx, wy, _FLOOR)
                self._set_cell(nx, ny, _FLOOR)
                stack.append((nx, ny))
            else:
                stack.pop()

        # Open some extra passages for more connectivity
        num_extra = int(self.rng.integers(8, 16))
        for _ in range(num_extra):
            x = int(self.rng.integers(1, w - 1))
            y = int(self.rng.integers(1, h - 1))
            if self._world_at(x, y) == "#":
                # Only open if it connects two floor cells
                adj_floor = 0
                for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if self._world_at(x + ddx, y + ddy) in (_FLOOR, _PELLET, _POWER):
                        adj_floor += 1
                if adj_floor >= 2:
                    self._set_cell(x, y, _FLOOR)

    def _advance_entities(self) -> float:
        """Move ghosts toward or away from agent."""
        if self._power_timer > 0:
            self._power_timer -= 1
            if self._power_timer == 0:
                # Ghosts return to normal
                for e in self._entities:
                    if e.alive and e.etype == "ghost":
                        e.data["scared"] = False
                        e.char = "E"

        for e in self._entities:
            if not e.alive or e.etype != "ghost":
                continue
            # Move toward agent (chase) or away (flee)
            scared = e.data.get("scared", False)
            best_dir = self._pick_ghost_dir(e, flee=scared)
            if best_dir is not None:
                e.x += best_dir[0]
                e.y += best_dir[1]
        return 0.0

    def _pick_ghost_dir(
        self, ghost: Entity, flee: bool
    ) -> tuple[int, int] | None:
        """Pick a direction for the ghost to move."""
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        valid: list[tuple[int, int, float]] = []
        for ddx, ddy in dirs:
            nx, ny = ghost.x + ddx, ghost.y + ddy
            if self._world_at(nx, ny) != "#":
                dist = abs(nx - self._agent_x) + abs(ny - self._agent_y)
                valid.append((ddx, ddy, dist))
        if not valid:
            return None
        if flee:
            # Move away: pick direction that maximizes distance
            valid.sort(key=lambda t: -t[2])
        else:
            # Chase: pick direction that minimizes distance
            valid.sort(key=lambda t: t[2])
        # Small chance of random move for variety
        if len(valid) > 1 and float(self.rng.random()) < 0.3:
            idx = int(self.rng.integers(0, len(valid)))
            return (valid[idx][0], valid[idx][1])
        return (valid[0][0], valid[0][1])

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        if action_name == "LEFT":
            self._try_move(-1, 0)
        elif action_name == "RIGHT":
            self._try_move(1, 0)
        elif action_name == "UP":
            self._try_move(0, -1)
        elif action_name == "DOWN":
            self._try_move(0, 1)

        # Check pellet collection
        cell = self._world_at(self._agent_x, self._agent_y)
        if cell == _PELLET:
            self._set_cell(self._agent_x, self._agent_y, _FLOOR)
            reward += 0.5
            self._pellet_count -= 1
        elif cell == _POWER:
            self._set_cell(self._agent_x, self._agent_y, _FLOOR)
            reward += 0.5
            self._power_timer = 15  # 15 steps of power
            for e in self._entities:
                if e.alive and e.etype == "ghost":
                    e.data["scared"] = True
                    e.char = "F"  # frightened
            self._message = "Power pellet! Ghosts are scared!"

        # Check ghost collisions
        for e in self._entities:
            if not e.alive or e.etype != "ghost":
                continue
            if e.x == self._agent_x and e.y == self._agent_y:
                if e.data.get("scared", False):
                    # Eat ghost
                    e.alive = False
                    reward += 2.0
                    self._enemies_eaten += 1
                    self._message = "Ate a ghost!"
                else:
                    # Ghost kills agent
                    terminated = True
                    self._message = "Caught by a ghost!"
                    return reward, terminated, info

        # Check level clear
        if self._pellet_count <= 0:
            reward += 10.0
            terminated = True
            self._message = "Level cleared!"

        info["pellets_remaining"] = self._pellet_count
        info["power_timer"] = self._power_timer
        info["enemies_eaten"] = self._enemies_eaten
        return reward, terminated, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        if self._power_timer > 0:
            pwr = f"Power: {self._power_timer} turns"
        else:
            pwr = "Power: OFF"
        extra = (
            f"{pwr}  Pellets: {self._pellet_count}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "Collect all pellets (.) in the maze. Ghosts (E) chase you -- "
            "touching one kills you. Eat a power pellet (O) to make ghosts "
            "scared (F) for 15 steps, allowing you to eat them for +2. "
            "+0.5 per pellet. +10 for clearing the level."
        )

    def _symbol_meaning(self, ch: str) -> str:
        meanings = {
            " ": "empty floor",
            ".": "pellet (+0.5)",
            "#": "wall",
            "O": "power pellet",
            "E": "ghost (deadly)",
            "F": "scared ghost (edible)",
        }
        return meanings.get(ch, super()._symbol_meaning(ch))
