"""ProcgenBase: shared base for all Procgen game environments.

Provides world grid, agent-centered view rendering, entity management,
and basic gravity/physics support.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from atlas_rl.core.action import ActionSpec
from atlas_rl.core.ascii_primitives import build_legend, grid_to_string
from atlas_rl.core.base_env import BaseAsciiEnv
from atlas_rl.core.observation import GridObservation

VIEW_WIDTH = 20
VIEW_HEIGHT = 12

JUMP_ARC_DY: tuple[int, ...] = (-1, -1, -1, 0, 1, 1, 1)


@dataclass
class Entity:
    """A moving object in the world (enemy, item, projectile, etc.)."""

    etype: str  # entity type identifier
    char: str  # render character
    x: int
    y: int
    dx: int = 0  # velocity x (cells per step)
    dy: int = 0  # velocity y (cells per step)
    alive: bool = True
    data: dict[str, Any] = field(default_factory=dict)


class ProcgenBase(BaseAsciiEnv):
    """Abstract base for all Procgen environments.

    Subclasses MUST implement:
      - env_id() -> str
      - action_spec class attribute
      - _generate_level(seed) -> None
      - _game_step(action_name) -> tuple[float, bool, dict]
        Returns (reward, terminated, info_extras).
    """

    action_spec: ActionSpec  # must be set by subclass
    noop_action_name: str = "NOOP"

    def __init__(self, max_turns: int = 512) -> None:
        super().__init__(max_turns=max_turns)
        self._world: list[list[str]] = []
        self._world_w: int = 0
        self._world_h: int = 0
        self._agent_x: int = 0
        self._agent_y: int = 0
        self._entities: list[Entity] = []
        self._score: float = 0.0
        self._message: str = ""
        self._has_gravity: bool = False
        self._on_ground: bool = True
        self._jump_step: int = -1  # -1 = not jumping
        self._view_w: int = VIEW_WIDTH
        self._view_h: int = VIEW_HEIGHT

    # --- World setup helpers ---

    def _init_world(self, width: int, height: int, fill: str = ".") -> None:
        self._world_w = width
        self._world_h = height
        self._world = [[fill for _ in range(width)] for _ in range(height)]

    def _world_at(self, x: int, y: int) -> str:
        if 0 <= x < self._world_w and 0 <= y < self._world_h:
            return self._world[y][x]
        return "#"  # out of bounds = wall

    def _set_cell(self, x: int, y: int, ch: str) -> None:
        if 0 <= x < self._world_w and 0 <= y < self._world_h:
            self._world[y][x] = ch

    def _is_solid(self, x: int, y: int) -> bool:
        ch = self._world_at(x, y)
        return ch in ("#", "=", "+", "|", "-")

    def _add_entity(
        self, etype: str, char: str, x: int, y: int, **kwargs: Any
    ) -> Entity:
        e = Entity(etype=etype, char=char, x=x, y=y, **kwargs)
        self._entities.append(e)
        return e

    # --- Core loop ---

    def _reset(self, seed: int) -> GridObservation:
        self._entities = []
        self._score = 0.0
        self._message = ""
        self._jump_step = -1
        self._on_ground = True
        self._generate_level(seed)
        return self._render_current_observation()

    @abstractmethod
    def _generate_level(self, seed: int) -> None: ...

    @abstractmethod
    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        """Process one game tick. Returns (reward, terminated, extra_info)."""
        ...

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        self._message = ""

        reward, terminated, info = self._game_step(name)
        self._score += reward

        # Advance entities
        self._advance_entities()

        # Gravity (if enabled)
        if self._has_gravity and not terminated:
            self._apply_gravity()

        info["score"] = self._score
        info["agent_pos"] = (self._agent_x, self._agent_y)
        return self._render_current_observation(), reward, terminated, False, info

    def _advance_entities(self) -> None:
        """Move all entities by their velocity. Remove dead ones."""
        for e in self._entities:
            if not e.alive:
                continue
            e.x += e.dx
            e.y += e.dy
            # Remove if out of world bounds
            if (
                e.x < 0
                or e.x >= self._world_w
                or e.y < 0
                or e.y >= self._world_h
            ):
                e.alive = False
        self._entities = [e for e in self._entities if e.alive]

    def _apply_gravity(self) -> None:
        """Simple gravity: agent falls if not on solid ground and not jumping."""
        if self._jump_step >= 0:
            return  # jumping overrides gravity
        below = self._world_at(self._agent_x, self._agent_y + 1)
        if below not in ("#", "=", "+") and self._agent_y + 1 < self._world_h:
            # Check for entity below
            entity_below = any(
                e.x == self._agent_x and e.y == self._agent_y + 1
                for e in self._entities
                if e.alive and e.etype == "platform"
            )
            if not entity_below:
                self._agent_y += 1
                self._on_ground = False
        else:
            self._on_ground = True

    def _try_move(self, dx: int, dy: int) -> bool:
        """Try to move agent. Returns True if successful."""
        nx, ny = self._agent_x + dx, self._agent_y + dy
        if not self._is_solid(nx, ny):
            self._agent_x = nx
            self._agent_y = ny
            return True
        return False

    def _start_jump(self) -> None:
        """Start a jump arc if on ground."""
        if self._on_ground:
            self._jump_step = 0
            self._on_ground = False

    def _process_jump(self) -> None:
        """Advance jump arc by one step."""
        if self._jump_step < 0:
            return
        if self._jump_step < len(JUMP_ARC_DY):
            dy = JUMP_ARC_DY[self._jump_step]
            ny = self._agent_y + dy
            if dy < 0 and not self._is_solid(self._agent_x, ny):
                self._agent_y = ny
            elif dy > 0:
                if not self._is_solid(self._agent_x, ny):
                    self._agent_y = ny
                else:
                    self._on_ground = True
                    self._jump_step = -1
                    return
            self._jump_step += 1
        else:
            self._jump_step = -1

    # --- Rendering ---

    def _render_current_observation(self) -> GridObservation:
        # Compute view window
        vx0 = max(0, self._agent_x - self._view_w // 2)
        vy0 = max(0, self._agent_y - self._view_h // 2)
        vx1 = min(self._world_w, vx0 + self._view_w)
        vy1 = min(self._world_h, vy0 + self._view_h)
        vx0 = max(0, vx1 - self._view_w)
        vy0 = max(0, vy1 - self._view_h)

        actual_w = vx1 - vx0
        actual_h = vy1 - vy0

        render = [["." for _ in range(actual_w)] for _ in range(actual_h)]
        symbols: dict[str, str] = {}

        # Copy world tiles
        for ry in range(actual_h):
            for rx in range(actual_w):
                wx, wy = vx0 + rx, vy0 + ry
                ch = self._world_at(wx, wy)
                render[ry][rx] = ch
                if ch not in symbols:
                    symbols[ch] = self._symbol_meaning(ch)

        # Render entities in view
        for e in self._entities:
            if not e.alive:
                continue
            rx, ry = e.x - vx0, e.y - vy0
            if 0 <= rx < actual_w and 0 <= ry < actual_h:
                render[ry][rx] = e.char
                if e.char not in symbols:
                    symbols[e.char] = e.etype

        # Render agent
        arx, ary = self._agent_x - vx0, self._agent_y - vy0
        if 0 <= arx < actual_w and 0 <= ary < actual_h:
            render[ary][arx] = "@"
        symbols["@"] = "you"

        hud = (
            f"Score: {self._score:.0f}    "
            f"Turn: {self._turn}    "
            f"Pos: ({self._agent_x},{self._agent_y})"
        )

        return GridObservation(
            grid=grid_to_string(render),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _symbol_meaning(self, ch: str) -> str:
        """Override per-game for custom symbol meanings."""
        meanings = {
            ".": "empty",
            "=": "ground",
            "#": "wall",
            "|": "wall",
            "-": "wall",
            "+": "wall corner",
            "~": "water",
            "C": "coin",
            "$": "gold",
            "*": "star",
            "%": "fruit",
        }
        return meanings.get(ch, ch)

    def _task_description(self) -> str:
        return "Complete the level."

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            f"TASK\n{self._task_description()}\n\n"
            f"VIEW\n"
            f"You see a {self._view_w}x{self._view_h} window centered on your position. "
            f"The full level extends beyond the visible area.\n\n"
            + self.action_spec.render_for_prompt()
        )
