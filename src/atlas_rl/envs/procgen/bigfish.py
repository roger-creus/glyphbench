"""Procgen BigFish environment.

Ocean grid where the agent fish eats smaller fish to grow. Bigger fish
eat the agent.

Gym ID: atlas_rl/procgen-bigfish-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec
from atlas_rl.core.observation import GridObservation
from atlas_rl.envs.procgen.base import ProcgenBase

_WATER = "~"

# Fish size -> display char
_SIZE_CHARS = {1: "f", 2: "f", 3: "F", 4: "F", 5: "W", 6: "W"}


def _fish_char(size: int) -> str:
    if size <= 2:
        return "f"
    if size <= 4:
        return "F"
    return "W"


class BigFishEnv(ProcgenBase):
    """Procgen BigFish: eat smaller fish, avoid bigger ones."""

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

    GRID_W = 20
    GRID_H = 12

    def env_id(self) -> str:
        return "atlas_rl/procgen-bigfish-v0"

    def _generate_level(self, seed: int) -> None:
        w, h = self.GRID_W, self.GRID_H
        self._init_world(w, h, fill=_WATER)

        # Agent starts at left side, size 1
        self._agent_x = 1
        self._agent_y = h // 2
        self._agent_size = 1
        self._fish_eaten = 0

        # Spawn 8-12 fish of varying sizes
        num_fish = int(self.rng.integers(8, 13))
        for _ in range(num_fish):
            self._spawn_fish()

    def _spawn_fish(self, *, force_edge: bool = False) -> None:
        """Spawn a fish at a random position."""
        w, h = self.GRID_W, self.GRID_H
        size = int(self.rng.integers(1, 7))  # 1-6
        ch = _fish_char(size)

        if force_edge:
            # Spawn at left or right edge
            if int(self.rng.integers(0, 2)) == 0:
                x = 0
                dx = 1
            else:
                x = w - 1
                dx = -1
        else:
            x = int(self.rng.integers(3, w - 1))
            dx = 1 if int(self.rng.integers(0, 2)) == 0 else -1

        y = int(self.rng.integers(0, h))

        self._add_entity(
            "fish", ch, x, y, dx=dx,
            data={"size": size},
        )

    def _advance_entities(self) -> float:
        """Move fish horizontally. Wrap around edges. Respawn eaten fish."""
        for e in self._entities:
            if not e.alive:
                continue
            e.x += e.dx
            # Wrap around
            if e.x < 0:
                e.x = self.GRID_W - 1
                e.dx = 1
            elif e.x >= self.GRID_W:
                e.x = 0
                e.dx = -1

            # Small random vertical drift
            if float(self.rng.random()) < 0.2:
                dy = 1 if int(self.rng.integers(0, 2)) == 0 else -1
                ny = e.y + dy
                if 0 <= ny < self.GRID_H:
                    e.y = ny

        # Maintain minimum fish count
        alive_count = sum(1 for e in self._entities if e.alive)
        while alive_count < 6:
            self._spawn_fish(force_edge=True)
            alive_count += 1
        return 0.0

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        # Movement (open ocean, no walls)
        dx, dy = 0, 0
        if action_name == "LEFT":
            dx = -1
        elif action_name == "RIGHT":
            dx = 1
        elif action_name == "UP":
            dy = -1
        elif action_name == "DOWN":
            dy = 1

        if dx != 0 or dy != 0:
            self._agent_dir = (dx, dy)

        nx = self._agent_x + dx
        ny = self._agent_y + dy
        if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H:
            self._agent_x = nx
            self._agent_y = ny

        # Check fish collisions
        for e in self._entities:
            if not e.alive:
                continue
            if e.x == self._agent_x and e.y == self._agent_y:
                fish_size = e.data.get("size", 1)
                if fish_size <= self._agent_size:
                    # Eat smaller/equal fish
                    e.alive = False
                    reward += 1.0
                    self._fish_eaten += 1
                    # Grow every 3 fish eaten
                    if self._fish_eaten % 3 == 0 and self._agent_size < 6:
                        self._agent_size += 1
                        self._message = f"You grew to size {self._agent_size}!"
                else:
                    # Eaten by bigger fish
                    reward = -1.0
                    terminated = True
                    self._message = "Eaten by a bigger fish!"
                    break

        # Clean up dead entities
        self._entities = [e for e in self._entities if e.alive]

        info["agent_size"] = self._agent_size
        info["fish_eaten"] = self._fish_eaten
        return reward, terminated, info

    def _render_current_observation(self) -> GridObservation:
        obs = super()._render_current_observation()
        sz = self._agent_size
        extra = (
            f"Size: {sz}  Edible: size<={sz}"
            f"  Deadly: size>{sz}"
            f"  Eaten: {self._fish_eaten}"
        )
        new_hud = obs.hud + "\n" + extra
        return GridObservation(
            grid=obs.grid, legend=obs.legend,
            hud=new_hud, message=obs.message,
        )

    def _task_description(self) -> str:
        return (
            "You are a fish (@) in the ocean. Eat smaller fish (f) to grow. "
            "Avoid bigger fish (F, W) or they eat you. +1 per fish eaten, "
            "-1 on death. You grow every 3 fish eaten."
        )

    def _symbol_meaning(self, ch: str) -> str:
        meanings = {
            "~": "water",
            "f": "small fish",
            "F": "medium fish",
            "W": "large fish",
        }
        return meanings.get(ch, super()._symbol_meaning(ch))
