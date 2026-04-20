"""Atari Ms. Pac-Man environment.

A 28x31 maze. Player eats pellets and power pellets while avoiding ghosts.
Power pellets make ghosts frightened and edible.

Gym ID: atlas_rl/atari-mspacman-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec
from atlas_rl.core.observation import GridObservation

from .base import AtariBase, AtariEntity

# Direction vectors
_DIRS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

# Classic maze template (28 wide x 31 tall)
# '#' = wall, '.' = pellet, '*' = power pellet, ' ' = empty corridor
_MAZE_TEMPLATE = [
    "############################",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#*####.#####.##.#####.####*#",
    "#..........................#",
    "#.####.##.########.##.####.#",
    "#......##....##....##......#",
    "######.##### ## #####.######",
    "     #.##          ##.#     ",
    "     #.## ###--### ##.#     ",
    "######.## #      # ##.######",
    "      .   #      #   .      ",
    "######.## #      # ##.######",
    "     #.## ######## ##.#     ",
    "     #.##          ##.#     ",
    "######.## ######## ##.######",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#*..##.......  .......##..*#",
    "###.##.##.########.##.##.###",
    "#......##....##....##......#",
    "#.##########.##.##########.#",
    "#..........................#",
    "############################",
]

_MAZE_W = 28
_MAZE_H = len(_MAZE_TEMPLATE)

# Ghost spawn positions (inside the ghost pen)
_GHOST_SPAWNS = [(12, 11), (13, 11), (14, 11), (15, 11)]
_GHOST_CHARS = ["R", "P", "B", "O"]  # red, pink, blue, orange
_PLAYER_START = (14, 18)

_FRIGHTENED_DURATION = 20


class MsPacManEnv(AtariBase):
    """Ms. Pac-Man: eat pellets, avoid ghosts, use power pellets.

    Actions: NOOP, UP, RIGHT, LEFT, DOWN
    Reward: +10 per pellet, +50 per power pellet, +200 per ghost eaten.
    Lives: 3, level complete when all pellets eaten.
    """

    action_spec = ActionSpec(
        names=("NOOP", "UP", "RIGHT", "LEFT", "DOWN"),
        descriptions=(
            "do nothing this step",
            "move up one cell",
            "move right one cell",
            "move left one cell",
            "move down one cell",
        ),
    )

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._pellet_count: int = 0
        self._frightened_timer: int = 0
        self._ghost_eat_combo: int = 0
        self._player_dir: tuple[int, int] = (0, 0)

    def env_id(self) -> str:
        return "atlas_rl/atari-mspacman-v0"

    def _task_description(self) -> str:
        return (
            "Eat all pellets (.) and power pellets (*) to clear the level. "
            "Avoid ghosts (R, P, B, O) or eat a power pellet to make them "
            "frightened (F) and edible. Eating ghosts gives bonus points."
        )

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "#": "wall",
            ".": "pellet",
            "*": "power pellet",
            " ": "empty",
            "-": "ghost door",
        }.get(ch, ch)

    def _generate_level(self, seed: int) -> None:
        self._init_grid(_MAZE_W, _MAZE_H)
        self._pellet_count = 0
        self._frightened_timer = 0
        self._ghost_eat_combo = 0
        self._player_dir = (0, 0)
        self._entities = []

        # Build maze from template
        for y, row in enumerate(_MAZE_TEMPLATE):
            for x, ch in enumerate(row):
                self._set_cell(x, y, ch)
                if ch in (".", "*"):
                    self._pellet_count += 1

        # Place player
        self._player_x, self._player_y = _PLAYER_START

        # Spawn ghosts
        for i, (gx, gy) in enumerate(_GHOST_SPAWNS):
            ghost = self._add_entity(
                etype="ghost",
                char=_GHOST_CHARS[i],
                x=gx,
                y=gy,
            )
            ghost.data["home_x"] = gx
            ghost.data["home_y"] = gy
            ghost.data["color"] = _GHOST_CHARS[i]
            ghost.data["state"] = "scatter"  # chase / scatter / frightened
            ghost.data["dir"] = (0, -1)

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Tick frightened timer
        if self._frightened_timer > 0:
            self._frightened_timer -= 1
            if self._frightened_timer == 0:
                self._ghost_eat_combo = 0
                for e in self._entities:
                    if e.etype == "ghost" and e.data.get("state") == "frightened":
                        e.data["state"] = "chase"
                        e.char = e.data["color"]

        # Move player
        if action_name in _DIRS:
            dx, dy = _DIRS[action_name]
            self._player_dir = (dx, dy)
        else:
            dx, dy = 0, 0

        nx, ny = self._player_x + dx, self._player_y + dy
        # Tunnel wrap
        if nx < 0:
            nx = _MAZE_W - 1
        elif nx >= _MAZE_W:
            nx = 0
        if not self._is_solid(nx, ny) and self._grid_at(nx, ny) != "-":
            self._player_x, self._player_y = nx, ny

        # Check pellet collection
        cell = self._grid_at(self._player_x, self._player_y)
        if cell == ".":
            self._set_cell(self._player_x, self._player_y, " ")
            self._on_point_scored(10)
            reward += 10.0
            self._pellet_count -= 1
        elif cell == "*":
            self._set_cell(self._player_x, self._player_y, " ")
            self._on_point_scored(50)
            reward += 50.0
            self._pellet_count -= 1
            # Frighten ghosts
            self._frightened_timer = _FRIGHTENED_DURATION
            self._ghost_eat_combo = 0
            for e in self._entities:
                if e.etype == "ghost" and e.alive:
                    e.data["state"] = "frightened"
                    e.char = "F"

        # Move ghosts
        for e in self._entities:
            if e.etype != "ghost" or not e.alive:
                continue
            self._move_ghost(e)

        # Check ghost collisions
        for e in self._entities:
            if e.etype != "ghost" or not e.alive:
                continue
            if e.x == self._player_x and e.y == self._player_y:
                if e.data.get("state") == "frightened":
                    self._ghost_eat_combo += 1
                    pts = 200 * self._ghost_eat_combo
                    self._on_point_scored(pts)
                    reward += float(pts)
                    # Respawn ghost
                    e.x = e.data["home_x"]
                    e.y = e.data["home_y"]
                    e.data["state"] = "chase"
                    e.char = e.data["color"]
                else:
                    self._on_life_lost()
                    reward -= 100.0
                    if not self._game_over:
                        # Reset positions
                        self._player_x, self._player_y = _PLAYER_START
                        for g in self._entities:
                            if g.etype == "ghost":
                                g.x = g.data["home_x"]
                                g.y = g.data["home_y"]
                                g.data["state"] = "scatter"
                                g.char = g.data["color"]
                        self._frightened_timer = 0

        # Level complete?
        if self._pellet_count <= 0:
            self._on_point_scored(500)
            reward += 500.0
            self._message = "Level complete! +500"
            self._level += 1
            self._generate_level(self._level)

        info["pellets_remaining"] = self._pellet_count
        info["frightened_timer"] = self._frightened_timer
        return reward, self._game_over, info

    def _move_ghost(self, ghost: AtariEntity) -> None:
        """Simple ghost AI: chase player or move randomly."""
        state = ghost.data.get("state", "chase")
        # Get possible directions (exclude reversing)
        cur_dir = ghost.data.get("dir", (0, -1))
        reverse = (-cur_dir[0], -cur_dir[1])
        possible: list[tuple[int, int]] = []
        for d in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            if d == reverse:
                continue
            nx, ny = ghost.x + d[0], ghost.y + d[1]
            # Tunnel wrap
            if nx < 0:
                nx = _MAZE_W - 1
            elif nx >= _MAZE_W:
                nx = 0
            if not self._is_solid(nx, ny) and self._grid_at(nx, ny) != "-":
                possible.append(d)
        if not possible:
            # Allow reverse if stuck
            for d in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = ghost.x + d[0], ghost.y + d[1]
                if nx < 0:
                    nx = _MAZE_W - 1
                elif nx >= _MAZE_W:
                    nx = 0
                if not self._is_solid(nx, ny):
                    possible.append(d)
        if not possible:
            return

        if state == "frightened":
            chosen = possible[int(self.rng.integers(0, len(possible)))]
        elif state == "chase":
            # Move toward player
            best_dist = float("inf")
            chosen = possible[0]
            for d in possible:
                nx, ny = ghost.x + d[0], ghost.y + d[1]
                dist = abs(nx - self._player_x) + abs(ny - self._player_y)
                if dist < best_dist:
                    best_dist = dist
                    chosen = d
        else:
            # scatter: random
            chosen = possible[int(self.rng.integers(0, len(possible)))]

        nx, ny = ghost.x + chosen[0], ghost.y + chosen[1]
        if nx < 0:
            nx = _MAZE_W - 1
        elif nx >= _MAZE_W:
            nx = 0
        ghost.x, ghost.y = nx, ny
        ghost.data["dir"] = chosen

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        pwr = (
            str(self._frightened_timer)
            if self._frightened_timer > 0 else "OFF"
        )
        ghosts = []
        for e in self._entities:
            if e.etype == "ghost" and e.alive:
                st = e.data.get("state", "chase")
                ghosts.append(
                    f"{e.data['color']}={st}"
                )
        glist = ",".join(ghosts) if ghosts else "none"
        extra = (
            f"Pellets: {self._pellet_count}"
            f"  Power: {pwr}"
            f"  Ghosts: {glist}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _advance_entities(self) -> None:
        """Override: ghosts are moved in _game_step, skip default movement."""
        pass
