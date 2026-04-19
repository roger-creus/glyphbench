"""Atari Space Invaders environment.

Classic alien shooter. Rows of aliens march side-to-side and descend.

Gym ID: atlas_rl/atari-spaceinvaders-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec

from .base import AtariBase, AtariEntity


class SpaceInvadersEnv(AtariBase):
    """Space Invaders: shoot descending aliens.

    30x20 grid. 5 rows of 11 aliens, agent at bottom. Shields for cover.
    Aliens march side-to-side and step down when hitting a wall.

    Actions: NOOP, FIRE, LEFT, RIGHT, LEFT_FIRE, RIGHT_FIRE
    Reward: +10/+20/+30 per alien (by row), +100 mystery ship
    Lives: 3
    """

    action_spec = ActionSpec(
        names=("NOOP", "FIRE", "LEFT", "RIGHT", "LEFT_FIRE", "RIGHT_FIRE"),
        descriptions=(
            "do nothing",
            "fire a bullet upward",
            "move left",
            "move right",
            "move left and fire",
            "move right and fire",
        ),
    )

    _WIDTH = 30
    _HEIGHT = 20
    _PLAYER_Y = 18
    _ALIEN_ROWS = 5
    _ALIEN_COLS = 11
    _ALIEN_START_Y = 3
    _MYSTERY_INTERVAL = 50
    _ROW_SCORES = (30, 20, 20, 10, 10)  # top to bottom

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._aliens: list[list[AtariEntity | None]] = []
        self._alien_dir: int = 1  # 1=right, -1=left
        self._bullets: list[AtariEntity] = []
        self._enemy_bullets: list[AtariEntity] = []
        self._step_counter: int = 0
        self._alien_move_timer: int = 0
        self._alien_move_interval: int = 4
        self._shields: set[tuple[int, int]] = set()

    def env_id(self) -> str:
        return "atlas_rl/atari-spaceinvaders-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._bullets = []
        self._enemy_bullets = []
        self._step_counter = 0
        self._alien_move_timer = 0
        self._alien_dir = 1

        # Borders
        for x in range(self._WIDTH):
            self._set_cell(x, 0, "-")
            self._set_cell(x, self._HEIGHT - 1, "-")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "|")
            self._set_cell(self._WIDTH - 1, y, "|")

        # Create alien grid
        self._aliens = []
        chars = ("W", "V", "V", "M", "M")  # different shapes per row
        for row in range(self._ALIEN_ROWS):
            alien_row: list[AtariEntity | None] = []
            y = self._ALIEN_START_Y + row
            for col in range(self._ALIEN_COLS):
                x = 3 + col * 2
                if x < self._WIDTH - 1:
                    a = self._add_entity("alien", chars[row], x, y)
                    a.data["row"] = row
                    a.data["col"] = col
                    alien_row.append(a)
                else:
                    alien_row.append(None)
            self._aliens.append(alien_row)

        # Create shields
        self._shields = set()
        shield_positions = [6, 14, 22]
        for sx in shield_positions:
            for dx in range(3):
                for dy in range(2):
                    px = sx + dx
                    py = self._PLAYER_Y - 3 + dy
                    if 0 < px < self._WIDTH - 1 and 0 < py < self._HEIGHT - 1:
                        self._shields.add((px, py))

        # Player position
        self._player_x = self._WIDTH // 2
        self._player_y = self._PLAYER_Y

        self._redraw_field()

    def _game_step(self, action_name: str) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        # Move player
        wants_left = action_name in ("LEFT", "LEFT_FIRE")
        wants_right = action_name in ("RIGHT", "RIGHT_FIRE")
        wants_fire = action_name in ("FIRE", "LEFT_FIRE", "RIGHT_FIRE")

        if wants_left and self._player_x > 1:
            self._player_x -= 1
        elif wants_right and self._player_x < self._WIDTH - 2:
            self._player_x += 1

        # Fire bullet (max 1 on screen)
        if wants_fire and len(self._bullets) == 0:
            b = self._add_entity("bullet", "!", self._player_x, self._player_y - 1,
                                 dy=-1)
            self._bullets.append(b)

        # Move player bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.y += b.dy
            if b.y <= 0:
                b.alive = False
                continue
            # Check alien hit
            hit = False
            for row_idx, alien_row in enumerate(self._aliens):
                for col_idx, alien in enumerate(alien_row):
                    if alien is not None and alien.alive and alien.x == b.x and alien.y == b.y:
                        alien.alive = False
                        self._aliens[row_idx][col_idx] = None
                        b.alive = False
                        pts = self._ROW_SCORES[row_idx] if row_idx < len(self._ROW_SCORES) else 10
                        self._on_point_scored(pts)
                        reward += pts
                        self._message = f"Alien hit! +{pts}"
                        hit = True
                        break
                if hit:
                    break
            # Check shield hit
            if not hit and b.alive and (b.x, b.y) in self._shields:
                self._shields.discard((b.x, b.y))
                b.alive = False

        # Check mystery ship hit by bullets
        for b in self._bullets:
            if not b.alive:
                continue
            for e in self._entities:
                if e.alive and e.etype == "mystery" and e.x == b.x and e.y == b.y:
                    e.alive = False
                    b.alive = False
                    self._on_point_scored(100)
                    reward += 100
                    self._message = "Mystery ship! +100"

        self._bullets = [b for b in self._bullets if b.alive]

        # Move aliens periodically
        self._alien_move_timer += 1
        if self._alien_move_timer >= self._alien_move_interval:
            self._alien_move_timer = 0
            self._move_aliens()

        # Enemy fire
        if self._step_counter % 6 == 0:
            self._enemy_fire()

        # Move enemy bullets
        for eb in self._enemy_bullets:
            if not eb.alive:
                continue
            eb.y += eb.dy
            if eb.y >= self._HEIGHT - 1:
                eb.alive = False
                continue
            # Hit player
            if eb.x == self._player_x and eb.y == self._player_y:
                eb.alive = False
                self._on_life_lost()
                self._message = "Hit! Lost a life."
                self._player_x = self._WIDTH // 2
            # Hit shield
            if eb.alive and (eb.x, eb.y) in self._shields:
                self._shields.discard((eb.x, eb.y))
                eb.alive = False

        self._enemy_bullets = [eb for eb in self._enemy_bullets if eb.alive]

        # Mystery ship
        if self._step_counter % self._MYSTERY_INTERVAL == 0:
            m = self._add_entity("mystery", "?", 1, 1, dx=1)
            m.data["timer"] = 0

        # Move mystery ships
        for e in self._entities:
            if e.etype == "mystery" and e.alive:
                e.x += e.dx
                if e.x >= self._WIDTH - 1 or e.x <= 0:
                    e.alive = False

        # Check level clear
        alive_count = sum(
            1 for row in self._aliens for a in row if a is not None and a.alive
        )
        if alive_count == 0:
            self._message = "Wave cleared!"
            self._level += 1
            self._generate_level(self._level)

        # Check if aliens reached player
        for alien_row in self._aliens:
            for alien in alien_row:
                if alien is not None and alien.alive and alien.y >= self._PLAYER_Y:
                    self._game_over = True
                    self._message = "Aliens reached you! Game Over!"
                    return reward, True, info

        self._redraw_field()
        info["aliens_left"] = alive_count
        return reward, self._game_over, info

    def _move_aliens(self) -> None:
        """Move alien formation side-to-side, stepping down at edges."""
        # Check if any alien at edge
        at_edge = False
        for alien_row in self._aliens:
            for alien in alien_row:
                if alien is not None and alien.alive:
                    nx = alien.x + self._alien_dir
                    if nx <= 1 or nx >= self._WIDTH - 2:
                        at_edge = True
                        break
            if at_edge:
                break

        if at_edge:
            self._alien_dir = -self._alien_dir
            for alien_row in self._aliens:
                for alien in alien_row:
                    if alien is not None and alien.alive:
                        alien.y += 1
        else:
            for alien_row in self._aliens:
                for alien in alien_row:
                    if alien is not None and alien.alive:
                        alien.x += self._alien_dir

    def _enemy_fire(self) -> None:
        """Random bottom alien fires a bullet."""
        # Find bottom-most alive alien in a random column
        alive_cols: list[AtariEntity] = []
        for col in range(self._ALIEN_COLS):
            for row in range(self._ALIEN_ROWS - 1, -1, -1):
                if row < len(self._aliens) and col < len(self._aliens[row]):
                    a = self._aliens[row][col]
                    if a is not None and a.alive:
                        alive_cols.append(a)
                        break

        if alive_cols and len(self._enemy_bullets) < 3:
            shooter = alive_cols[int(self.rng.integers(len(alive_cols)))]
            eb = self._add_entity("enemy_bullet", "v", shooter.x, shooter.y + 1,
                                  dy=1)
            self._enemy_bullets.append(eb)

    def _redraw_field(self) -> None:
        """Redraw the entire field."""
        # Clear interior
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")

        # Draw shields
        for sx, sy in self._shields:
            self._set_cell(sx, sy, "#")

        # Draw aliens
        for alien_row in self._aliens:
            for alien in alien_row:
                if alien is not None and alien.alive and 0 < alien.x < self._WIDTH - 1 and 0 < alien.y < self._HEIGHT - 1:
                    self._set_cell(alien.x, alien.y, alien.char)

        # Draw bullets
        for b in self._bullets:
            if b.alive and 0 < b.x < self._WIDTH - 1 and 0 < b.y < self._HEIGHT - 1:
                self._set_cell(b.x, b.y, "!")

        # Draw enemy bullets
        for eb in self._enemy_bullets:
            if eb.alive and 0 < eb.x < self._WIDTH - 1 and 0 < eb.y < self._HEIGHT - 1:
                self._set_cell(eb.x, eb.y, "v")

        # Draw mystery ships
        for e in self._entities:
            if e.etype == "mystery" and e.alive and 0 < e.x < self._WIDTH - 1 and 0 < e.y < self._HEIGHT - 1:
                self._set_cell(e.x, e.y, "?")

    def _advance_entities(self) -> None:
        # All movement handled in _game_step
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "-": "wall",
            "|": "wall",
            "W": "alien (top row, 30pts)",
            "V": "alien (mid rows, 20pts)",
            "M": "alien (bottom rows, 10pts)",
            "#": "shield",
            "!": "your bullet",
            "v": "enemy bullet",
            "?": "mystery ship (100pts)",
            " ": "empty",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Shoot the descending aliens before they reach you. "
            "Use shields for cover. Hit the mystery ship for bonus points."
        )
