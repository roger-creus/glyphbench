"""Atari H.E.R.O. environment.

Cave rescue game. Navigate mine shafts, use dynamite and laser.

Gym ID: glyphbench/atari-hero-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.action import ActionSpec

from .base import AtariBase


class HeroEnv(AtariBase):
    """H.E.R.O.: Helicopter Emergency Rescue Operation.

    Navigate mine shafts using a helicopter pack, dynamite, and laser.
    Rescue survivors at the bottom of the mine.

    Grid: 20x16.
    Actions: NOOP, FIRE, UP, RIGHT, LEFT, DOWN
    Reward: +75 per wall cleared, +1000 per survivor rescued
    """

    action_spec = ActionSpec(
        names=("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN"),
        descriptions=(
            "do nothing",
            "fire laser / place dynamite",
            "fly up (helicopter pack)",
            "move right",
            "move left",
            "move down",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 16
    _MAX_DYNAMITE = 6

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._dynamite: int = self._MAX_DYNAMITE
        self._walls: set[tuple[int, int]] = set()
        self._lava: set[tuple[int, int]] = set()
        self._survivor_pos: tuple[int, int] = (0, 0)
        self._survivor_alive: bool = True

    def env_id(self) -> str:
        return "glyphbench/atari-hero-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._dynamite = self._MAX_DYNAMITE
        self._walls = set()
        self._lava = set()
        self._survivor_alive = True

        # Border
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "─")
            self._set_cell(x, self._HEIGHT - 1, "─")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "│")
            self._set_cell(self._WIDTH - 1, y, "│")

        rng = self.rng

        # Generate mine shaft with platforms and passages
        # Create a series of horizontal platforms with gaps
        for shaft_y in range(4, self._HEIGHT - 2, 3):
            # Platform across most of the width
            gap_x = int(rng.integers(3, self._WIDTH - 4))
            gap_w = 2
            for x in range(1, self._WIDTH - 1):
                if gap_x <= x < gap_x + gap_w:
                    continue  # gap in platform
                self._set_cell(x, shaft_y, "█")

        # Place destructible walls (marked with 'X')
        wall_positions = [
            (int(rng.integers(3, self._WIDTH - 4)), 3),
            (int(rng.integers(3, self._WIDTH - 4)), 6),
            (int(rng.integers(3, self._WIDTH - 4)), 9),
            (int(rng.integers(3, self._WIDTH - 4)), 12),
        ]
        for wx, wy in wall_positions:
            if 0 < wx < self._WIDTH - 1 and 0 < wy < self._HEIGHT - 1:
                self._walls.add((wx, wy))
                self._set_cell(wx, wy, "X")

        # Place some lava
        for _ in range(2):
            lx = int(rng.integers(2, self._WIDTH - 3))
            ly = self._HEIGHT - 2
            self._lava.add((lx, ly))
            self._set_cell(lx, ly, "~")

        # Place enemies (bats)
        for _ in range(2 + self._level):
            ex = int(rng.integers(2, self._WIDTH - 3))
            ey = int(rng.integers(3, self._HEIGHT - 3))
            if self._grid_at(ex, ey) == " ":
                dx_choice = 1 if rng.random() < 0.5 else -1
                e = self._add_entity("bat", "b", ex, ey, dx=dx_choice)
                e.data["min_x"] = max(1, ex - 3)
                e.data["max_x"] = min(self._WIDTH - 2, ex + 3)

        # Survivor at bottom
        self._survivor_pos = (self._WIDTH // 2, self._HEIGHT - 2)
        sx, sy = self._survivor_pos
        self._set_cell(sx, sy, "S")

        # Player starts at top
        self._player_x = self._WIDTH // 2
        self._player_y = 1

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}

        dx, dy = 0, 0

        if action_name == "UP":
            dy = -1
            self._player_dir = (0, -1)
        elif action_name == "DOWN":
            dy = 1
            self._player_dir = (0, 1)
        elif action_name == "LEFT":
            dx = -1
            self._player_dir = (-1, 0)
        elif action_name == "RIGHT":
            dx = 1
            self._player_dir = (1, 0)
        elif action_name == "FIRE":
            # Fire laser ahead or place dynamite
            reward += self._use_fire()

        # Move player
        new_x = self._player_x + dx
        new_y = self._player_y + dy

        if 0 < new_x < self._WIDTH - 1 and 0 < new_y < self._HEIGHT - 1:
            cell = self._grid_at(new_x, new_y)
            if cell not in ("█", "X"):
                self._player_x = new_x
                self._player_y = new_y

        # Check lava
        if (self._player_x, self._player_y) in self._lava:
            self._on_life_lost()
            self._message = "Lava! Lost a life."
            self._player_x = self._WIDTH // 2
            self._player_y = 1

        # Check enemy collision
        for e in self._entities:
            if e.alive and e.etype == "bat" and e.x == self._player_x and e.y == self._player_y:
                self._on_life_lost()
                self._message = "Hit by bat! Lost a life."
                self._player_x = self._WIDTH // 2
                self._player_y = 1
                e.alive = False

        # Check survivor rescue
        if self._survivor_alive:
            sx, sy = self._survivor_pos
            if self._player_x == sx and self._player_y == sy:
                self._survivor_alive = False
                self._on_point_scored(1000)
                reward += 1000
                self._message = "Survivor rescued! +1000"
                self._level += 1
                self._generate_level(self._level + seed_offset(self._level))

        # Move enemies
        for e in self._entities:
            if e.alive and e.etype == "bat":
                e.x += e.dx
                min_x = e.data.get("min_x", 1)
                max_x = e.data.get("max_x", self._WIDTH - 2)
                if e.x <= min_x or e.x >= max_x:
                    e.dx = -e.dx
                if 0 < e.x < self._WIDTH - 1 and 0 < e.y < self._HEIGHT - 1:
                    pass
                else:
                    e.x -= e.dx

        # Redraw enemies on grid
        self._redraw_enemies()

        info["dynamite"] = self._dynamite
        info["survivor_alive"] = self._survivor_alive
        return reward, self._game_over, info

    def _use_fire(self) -> float:
        """Fire laser at adjacent walls or place dynamite."""
        reward = 0.0
        # Check all 4 adjacent cells for destructible walls
        for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            tx = self._player_x + ddx
            ty = self._player_y + ddy
            if (tx, ty) in self._walls:
                if self._dynamite > 0:
                    self._walls.discard((tx, ty))
                    self._set_cell(tx, ty, " ")
                    self._dynamite -= 1
                    self._on_point_scored(75)
                    reward += 75
                    self._message = f"Wall cleared! +75 (dynamite: {self._dynamite})"
                else:
                    self._message = "No dynamite left!"
                break

        # Kill adjacent bats with laser
        for e in self._entities:
            if (
                e.alive
                and e.etype == "bat"
                and abs(e.x - self._player_x) <= 1
                and abs(e.y - self._player_y) <= 1
            ):
                e.alive = False
                self._on_point_scored(50)
                reward += 50
                self._message = "Bat zapped! +50"
                break

        return reward

    def _redraw_enemies(self) -> None:
        """Clear and redraw enemy positions."""
        # Clear old enemy chars from grid
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                if self._grid_at(x, y) == "b":
                    self._set_cell(x, y, " ")

        for e in self._entities:
            if e.alive and 0 < e.x < self._WIDTH - 1 and 0 < e.y < self._HEIGHT - 1:
                self._set_cell(e.x, e.y, e.char)

        # Redraw survivor
        if self._survivor_alive:
            sx, sy = self._survivor_pos
            self._set_cell(sx, sy, "S")

    def _advance_entities(self) -> None:
        # Movement handled in _game_step
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "─": "border",
            "│": "border",
            "█": "rock wall",
            "X": "destructible wall (use dynamite)",
            "~": "lava",
            "S": "survivor (rescue target)",
            "b": "bat (enemy)",
            " ": "empty",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Navigate the mine shaft to rescue the survivor at the bottom. "
            "Use your helicopter pack to fly UP. "
            "FIRE to use laser on enemies or dynamite on walls. "
            "Avoid lava and bats."
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Atari H.E.R.O.\n\n"
            "TASK\n"
            "Descend a mine shaft with a helicopter pack and rescue "
            "a trapped survivor at the bottom. Use dynamite to blow "
            "up destructible walls and a short-range laser to kill "
            "bats. Reaching and touching the survivor advances the "
            "level to a new shaft.\n\n"
            "BOARD\n"
            "20x16 mine. Border walls. Solid rock platforms '#' cross "
            "the shaft every 3 rows with one 2-wide gap each. "
            "Destructible walls 'X' block parts of rows 3, 6, 9, 12. "
            "Lava 'tilde' cells appear on the bottom row. Bats 'b' "
            "patrol horizontally in a 7-cell range. Survivor 'S' "
            "sits near bottom-center. You are an arrow glyph.\n\n"
            "MECHANICS\n"
            "UP flies you 1 cell up; DOWN falls 1 cell; LEFT/RIGHT "
            "shift 1 cell. You cannot enter rock '#' or destructible "
            "'X' without clearing the latter first. FIRE: if the "
            "cell in any 4-adjacent direction contains an 'X' and "
            "you have dynamite, it blows the wall (uses 1 of 6 "
            "dynamites) and also zaps any adjacent bat.\n\n"
            "SCORING\n"
            "+75 reward per destructible wall cleared with dynamite. "
            "+50 reward per bat zapped with the laser/dynamite "
            "blast. +1000 reward for rescuing the survivor.\n\n"
            "TERMINATION\n"
            "Three lives. Stepping on lava or touching a bat costs a "
            "life and respawns you at the top. Episode ends at 0 "
            "lives or after max_turns.\n\n"
            "HUD\n"
            "Shows score, lives, level, dynamite remaining, and "
            "survivor status.\n\n"
            + self.action_spec.render_for_prompt()
        )


def seed_offset(level: int) -> int:
    """Deterministic offset for level regeneration."""
    return level * 1000
