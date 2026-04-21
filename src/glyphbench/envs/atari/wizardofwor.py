"""Atari Wizard of Wor environment.

Navigate a maze, shoot warrior ghosts. Clear all enemies to advance.

Gym ID: glyphbench/atari-wizardofwor-v0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation

from .base import AtariBase, AtariEntity

_DIRS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

_W = 20
_H = 16
_GHOST_CHAR = "w"
_BULLET_CHAR = "!"
_WIZARD_CHAR = "W"


class WizardOfWorEnv(AtariBase):
    """Wizard of Wor: shoot warrior ghosts in a maze.

    Actions: NOOP, FIRE, UP, RIGHT, LEFT, DOWN
    Reward: +100 per ghost, +500 for the Wizard.
    Level clears when all enemies are dead.
    """

    action_spec = ActionSpec(
        names=("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN"),
        descriptions=(
            "do nothing this step",
            "fire bullet in last-faced direction",
            "move up one cell",
            "move right one cell",
            "move left one cell",
            "move down one cell",
        ),
    )

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._facing: tuple[int, int] = (1, 0)
        self._enemies_killed: int = 0
        self._total_enemies: int = 0
        self._player_bullet_cooldown: int = 0

    def env_id(self) -> str:
        return "glyphbench/atari-wizardofwor-v0"

    def _task_description(self) -> str:
        return (
            "Shoot all warrior ghosts (w) and the Wizard (W) to clear the level. "
            "Navigate the maze corridors and use your bullets (!) wisely."
        )

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "█": "wall",
            " ": "empty corridor",
            "w": "warrior ghost",
            "W": "Wizard of Wor",
            "!": "bullet",
        }.get(ch, ch)

    def _generate_level(self, seed: int) -> None:
        rng = np.random.default_rng(seed + self._level * 1000)
        self._init_grid(_W, _H)
        self._entities = []
        self._enemies_killed = 0
        self._player_bullet_cooldown = 0
        self._facing = (1, 0)

        # Build maze: start with all walls, carve corridors
        for y in range(_H):
            for x in range(_W):
                self._set_cell(x, y, "█")

        # Border stays wall; carve interior corridors
        # Horizontal corridors
        corridor_rows = [2, 5, 8, 11, 13]
        for y in corridor_rows:
            if y < _H - 1:
                for x in range(1, _W - 1):
                    self._set_cell(x, y, " ")

        # Vertical corridors connecting horizontals
        n_verts = int(rng.integers(5, 9))
        for _ in range(n_verts):
            vx = int(rng.integers(2, _W - 2))
            start_row = int(rng.integers(0, len(corridor_rows) - 1))
            end_row = int(rng.integers(start_row + 1, len(corridor_rows)))
            y_start = corridor_rows[start_row]
            y_end = corridor_rows[end_row]
            for y in range(y_start, y_end + 1):
                if 0 < y < _H - 1:
                    self._set_cell(vx, y, " ")

        # Always ensure left and right edge corridors
        for y in range(1, _H - 1):
            if y in corridor_rows:
                self._set_cell(1, y, " ")
                self._set_cell(_W - 2, y, " ")

        # Place player
        self._player_x = 1
        self._player_y = corridor_rows[-1]

        # Place ghosts
        n_ghosts = min(6, 3 + self._level)
        self._total_enemies = 0
        for _ in range(n_ghosts):
            for _attempt in range(30):
                gx = int(rng.integers(2, _W - 2))
                gy_idx = int(rng.integers(0, len(corridor_rows)))
                gy = corridor_rows[gy_idx]
                if (
                    self._grid_at(gx, gy) == " "
                    and abs(gx - self._player_x) + abs(gy - self._player_y) > 5
                ):
                    ghost = self._add_entity("ghost", _GHOST_CHAR, gx, gy)
                    ghost.data["dir"] = (1, 0)
                    ghost.data["patrol_row"] = gy
                    self._total_enemies += 1
                    break

        # Place wizard on higher levels
        if self._level >= 2:
            for _attempt in range(30):
                wx = int(rng.integers(3, _W - 3))
                wy = corridor_rows[0]
                if self._grid_at(wx, wy) == " ":
                    wiz = self._add_entity("wizard", _WIZARD_CHAR, wx, wy)
                    wiz.data["dir"] = (1, 0)
                    wiz.data["teleport_timer"] = int(rng.integers(10, 25))
                    self._total_enemies += 1
                    break

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        # Bullet cooldown
        if self._player_bullet_cooldown > 0:
            self._player_bullet_cooldown -= 1

        # Parse action
        move_dir: tuple[int, int] | None = None
        fire = False
        if action_name == "FIRE":
            fire = True
        elif action_name in _DIRS:
            move_dir = _DIRS[action_name]
            self._facing = move_dir
            self._player_dir = move_dir

        # Move player
        if move_dir is not None:
            nx = self._player_x + move_dir[0]
            ny = self._player_y + move_dir[1]
            # Tunnel wrap on left/right edges
            if nx <= 0:
                nx = _W - 2
            elif nx >= _W - 1:
                nx = 1
            if not self._is_solid(nx, ny):
                self._player_x, self._player_y = nx, ny

        # Fire bullet
        if fire and self._player_bullet_cooldown <= 0:
            bx = self._player_x + self._facing[0]
            by = self._player_y + self._facing[1]
            if not self._is_solid(bx, by):
                b = self._add_entity("bullet", _BULLET_CHAR, bx, by)
                b.dx = self._facing[0]
                b.dy = self._facing[1]
                b.data["owner"] = "player"
                self._player_bullet_cooldown = 3

        # Move bullets and check collisions immediately
        for e in self._entities:
            if e.etype != "bullet" or not e.alive:
                continue
            e.x += e.dx
            e.y += e.dy
            if self._is_solid(e.x, e.y):
                e.alive = False

        # Check bullet-enemy collisions (before ghost movement so bullets
        # reliably hit stationary targets)
        for b in self._entities:
            if b.etype != "bullet" or not b.alive:
                continue
            if b.data.get("owner") != "player":
                # Enemy bullet: check player hit
                if b.x == self._player_x and b.y == self._player_y:
                    b.alive = False
                    self._on_life_lost()
                    reward -= 100.0
                    if not self._game_over:
                        self._player_x = 1
                        py_options = [2, 5, 8, 11, 13]
                        self._player_y = py_options[-1]
                continue
            for enemy in self._entities:
                if enemy.etype not in ("ghost", "wizard") or not enemy.alive:
                    continue
                if b.x == enemy.x and b.y == enemy.y:
                    b.alive = False
                    enemy.alive = False
                    self._enemies_killed += 1
                    pts = 500 if enemy.etype == "wizard" else 100
                    self._on_point_scored(pts)
                    reward += float(pts)

        # Move ghosts (after bullet collision checks)
        for e in self._entities:
            if e.etype == "ghost" and e.alive:
                self._move_ghost(e)
            elif e.etype == "wizard" and e.alive:
                self._move_wizard(e)

        # Check enemy-player collision
        for e in self._entities:
            if (
                e.etype in ("ghost", "wizard")
                and e.alive
                and e.x == self._player_x
                and e.y == self._player_y
            ):
                    self._on_life_lost()
                    reward -= 100.0
                    if not self._game_over:
                        self._player_x = 1
                        self._player_y = 13
                    break

        # Clean dead entities
        self._entities = [e for e in self._entities if e.alive]

        # Check level complete
        enemies_alive = sum(
            1 for e in self._entities if e.etype in ("ghost", "wizard") and e.alive
        )
        terminated = False
        if enemies_alive == 0 and self._total_enemies > 0:
            self._message = "Level complete!"
            self._level += 1
            self._generate_level(self._level * 3571)
            info["level_cleared"] = True

        info["enemies_alive"] = enemies_alive
        info["enemies_killed"] = self._enemies_killed
        return reward, terminated or self._game_over, info

    def _move_ghost(self, ghost: AtariEntity) -> None:
        """Ghost patrols horizontally, occasionally changing direction."""
        cur_dir = ghost.data.get("dir", (1, 0))
        if self.rng.random() < 0.1:
            # Change direction
            dirs = [(1, 0), (-1, 0), (0, -1), (0, 1)]
            new_dir = dirs[int(self.rng.integers(0, len(dirs)))]
            nx, ny = ghost.x + new_dir[0], ghost.y + new_dir[1]
            if not self._is_solid(nx, ny):
                cur_dir = new_dir
                ghost.data["dir"] = cur_dir

        nx, ny = ghost.x + cur_dir[0], ghost.y + cur_dir[1]
        # Wrap on edges
        if nx <= 0:
            nx = _W - 2
        elif nx >= _W - 1:
            nx = 1
        if not self._is_solid(nx, ny):
            ghost.x, ghost.y = nx, ny
        else:
            # Reverse
            ghost.data["dir"] = (-cur_dir[0], -cur_dir[1])

    def _move_wizard(self, wizard: AtariEntity) -> None:
        """Wizard can teleport and is faster."""
        wizard.data["teleport_timer"] = wizard.data.get("teleport_timer", 15) - 1
        if wizard.data["teleport_timer"] <= 0:
            wizard.data["teleport_timer"] = int(self.rng.integers(10, 20))
            # Teleport to a random corridor cell
            corridor_rows = [2, 5, 8, 11, 13]
            for _attempt in range(10):
                tx = int(self.rng.integers(2, _W - 2))
                ty = corridor_rows[int(self.rng.integers(0, len(corridor_rows)))]
                if not self._is_solid(tx, ty):
                    wizard.x, wizard.y = tx, ty
                    return

        # Otherwise move like a fast ghost
        self._move_ghost(wizard)

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        dname = self._DIR_NAMES.get(self._facing, "none")
        enemies = sum(
            1 for e in self._entities
            if e.etype in ("ghost", "wizard") and e.alive
        )
        extra = (
            f"Facing: {dname}  Enemies: {enemies}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _advance_entities(self) -> None:
        """Override: entities are moved in _game_step."""
        pass
