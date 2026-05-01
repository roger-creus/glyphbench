"""miniatari Beam Rider.

Identity: Ship rides 5 vertical beams shooting enemies that descend.
Win condition: clear 1 sector (5 enemies destroyed).
Reward: Pattern D, +1/5 per kill, -1 on enemy collision.
Loss: collision with descending enemy (terminal -1).

Gym ID: glyphbench/miniatari-beamrider-v0

Random baseline (seed=0..29): success_rate=40%, mean_length=30, mean_return=+0.120
"""
from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec
from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.miniatari.base import MiniatariBase


class MiniBeamRiderEnv(MiniatariBase):
    """Mini Beam Rider: 14x12 grid, 5 vertical beams.

    Beams are vertical guides at columns 2, 5, 7, 9, 12. The player ship
    snaps to one beam (LEFT/RIGHT cycles between beams). Enemies (E)
    spawn at row 0 on a random beam and descend at 1 cell every 2 ticks.
    FIRE shoots a torpedo straight up the player's beam, destroying the
    lowest enemy on that beam. Clear 5 enemies to win. Collision with
    an enemy (enemy reaches player row in player's beam) is -1 terminal.
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "snap to the next beam to the left",
            "snap to the next beam to the right",
            "fire a torpedo straight up your beam",
        ),
    )

    default_max_turns = 200

    _WIDTH = 14
    _HEIGHT = 12
    _WIN_TARGET = 5
    _BEAMS = (2, 5, 7, 9, 12)
    _PLAYER_Y = 11
    _ENEMY_MOVE_EVERY = 2
    _SPAWN_PROB = 0.6
    _FIRE_COOLDOWN = 3

    def __init__(self, max_turns: int | None = None) -> None:
        super().__init__(max_turns=max_turns)
        self._beam_idx: int = 0
        self._enemies: list[list[int]] = []  # [beam_idx, y]
        self._tick_count: int = 0
        self._progress: int = 0
        self._fire_cd: int = 0

    def env_id(self) -> str:
        return "glyphbench/miniatari-beamrider-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._progress = 0
        self._tick_count = 0
        self._fire_cd = 0
        self._enemies = []
        self._beam_idx = len(self._BEAMS) // 2
        self._player_x = self._BEAMS[self._beam_idx]
        self._player_y = self._PLAYER_Y
        # Pre-seed 2 enemies on distinct (beam, y) cells (no collisions).
        rng = self.rng
        used: set[tuple[int, int]] = set()
        for _ in range(2):
            for _attempt in range(40):
                bi = int(rng.integers(0, len(self._BEAMS)))
                y = int(rng.integers(0, 4))
                if (bi, y) in used:
                    continue
                used.add((bi, y))
                self._enemies.append([bi, y])
                break

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._tick_count += 1
        if self._fire_cd > 0:
            self._fire_cd -= 1

        # 1. Beam change
        if action_name == "LEFT" and self._beam_idx > 0:
            self._beam_idx -= 1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT" and self._beam_idx < len(self._BEAMS) - 1:
            self._beam_idx += 1
            self._player_dir = (1, 0)
        self._player_x = self._BEAMS[self._beam_idx]

        # 2. Fire (instantaneous beam shot)
        if action_name == "FIRE" and self._fire_cd == 0:
            self._fire_cd = self._FIRE_COOLDOWN
            target: int | None = None
            target_y: int = -1
            for i, (eb, ey) in enumerate(self._enemies):
                if eb == self._beam_idx and ey > target_y and ey < self._PLAYER_Y:
                    target = i
                    target_y = ey
            if target is not None:
                self._enemies.pop(target)
                reward += self._progress_reward(self._WIN_TARGET)
                self._progress += 1
                self._message = f"Enemy down! ({self._progress}/{self._WIN_TARGET})"
                if self._progress >= self._WIN_TARGET:
                    self._on_won()
                    return reward, self._game_over, info

        # 3. Enemies descend
        if self._tick_count % self._ENEMY_MOVE_EVERY == 0:
            survivors = []
            for e in self._enemies:
                e[1] += 1
                if e[1] < self._PLAYER_Y:
                    survivors.append(e)
                elif e[1] == self._PLAYER_Y and e[0] == self._beam_idx:
                    # Collision
                    self._message = "Enemy crashed into your ship!"
                    reward = self._death_reward()
                    self._on_life_lost()
                    return reward, True, info
                # else: enemy passed without hitting (different beam) -> remove silently
            self._enemies = survivors

        # 4. Spawn enemy
        rng = self.rng
        if rng.random() < self._SPAWN_PROB:
            # Pick a beam, avoid stacking at row 0
            occupied = {e[0] for e in self._enemies if e[1] == 0}
            free_beams = [i for i in range(len(self._BEAMS)) if i not in occupied]
            if free_beams:
                bi = free_beams[int(rng.integers(0, len(free_beams)))]
                self._enemies.append([bi, 0])

        info["progress"] = self._progress
        info["enemies_in_play"] = len(self._enemies)
        return reward, self._game_over, info

    def _render_current_observation(self) -> GridObservation:
        grid: list[list[str]] = [
            [" " for _ in range(self._WIDTH)] for _ in range(self._HEIGHT)
        ]
        # Beams
        for y in range(self._HEIGHT):
            for bx in self._BEAMS:
                grid[y][bx] = "│"
        # Enemies
        for eb, ey in self._enemies:
            x = self._BEAMS[eb]
            if 0 <= x < self._WIDTH and 0 <= ey < self._HEIGHT:
                grid[ey][x] = "E"
        # Player ship
        if 0 <= self._player_x < self._WIDTH and 0 <= self._player_y < self._HEIGHT:
            grid[self._player_y][self._player_x] = "Y"

        symbols = {
            " ": "void",
            "│": "beam",
            "E": "enemy",
            "Y": "your ship",
        }

        enemies_info = " ".join(
            f"beam{eb}@y={ey}" for eb, ey in self._enemies
        )
        hud = (
            f"Step: {self._turn} / {self.max_turns}    "
            f"Killed: {self._progress}/{self._WIN_TARGET}    "
            f"Score: {self._score:.3f}\n"
            f"On beam {self._beam_idx} (x={self._player_x})    "
            f"Enemies: {enemies_info}"
        )

        return GridObservation(
            grid=grid_to_string(grid),
            legend=build_legend(symbols),
            hud=hud,
            message=self._message,
        )

    def _task_description(self) -> str:
        return (
            "Mini Beam Rider on a 14x12 grid with 5 vertical beams (│) at "
            "columns 2, 5, 7, 9, 12. Your ship (Y) snaps to one beam at "
            "row 11. LEFT/RIGHT cycles to adjacent beams. FIRE shoots a "
            "torpedo straight up your beam and instantly destroys the "
            "lowest enemy on that beam. Enemies (E) spawn at the top on a "
            "random beam and descend 1 row every 2 ticks. Destroying 5 "
            "enemies wins the sector. If an enemy reaches your row in "
            "your beam, you take a -1 terminal penalty. Reward: +1/5 per "
            "kill, -1 on collision."
        )
