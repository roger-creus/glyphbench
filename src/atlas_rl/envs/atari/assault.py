"""Atari Assault environment.

Fixed turret at bottom, enemies descend in formations.

Gym ID: atlas_rl/atari-assault-v0
"""

from __future__ import annotations

from typing import Any

from atlas_rl.core.action import ActionSpec

from .base import AtariBase, AtariEntity


class AssaultEnv(AtariBase):
    """Assault: defend with a turret against descending enemies.

    20x20 grid. Enemies descend in formations. Player turret
    at bottom can move left/right and fire upward.

    Actions: NOOP, LEFT, RIGHT, FIRE
    Reward: +10 per enemy, +25 per formation leader
    Lives: 3
    """

    action_spec = ActionSpec(
        names=("NOOP", "LEFT", "RIGHT", "FIRE"),
        descriptions=(
            "do nothing",
            "move turret left",
            "move turret right",
            "fire a bullet upward",
        ),
    )

    _WIDTH = 20
    _HEIGHT = 20
    _PLAYER_Y = 18

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._enemies: list[AtariEntity] = []
        self._bullets: list[AtariEntity] = []
        self._enemy_bullets: list[AtariEntity] = []
        self._step_counter: int = 0
        self._spawn_timer: int = 0

    def env_id(self) -> str:
        return "atlas_rl/atari-assault-v0"

    def _generate_level(self, seed: int) -> None:
        self._init_grid(self._WIDTH, self._HEIGHT)
        self._entities = []
        self._enemies = []
        self._bullets = []
        self._enemy_bullets = []
        self._step_counter = 0
        self._spawn_timer = 0

        for x in range(self._WIDTH):
            self._set_cell(x, 0, "-")
            self._set_cell(x, self._HEIGHT - 1, "-")
        for y in range(self._HEIGHT):
            self._set_cell(0, y, "|")
            self._set_cell(self._WIDTH - 1, y, "|")

        self._player_x = self._WIDTH // 2
        self._player_y = self._PLAYER_Y

        self._spawn_formation()
        self._redraw()

    def _spawn_formation(self) -> None:
        rng = self.rng
        cx = int(rng.integers(4, self._WIDTH - 4))
        # Leader
        leader = self._add_entity("leader", "A", cx, 2)
        leader.data["dir"] = 1
        self._enemies.append(leader)
        # Wingmen
        for dx in (-2, 2):
            nx = cx + dx
            if 1 < nx < self._WIDTH - 1:
                e = self._add_entity("enemy", "a", nx, 3)
                e.data["dir"] = 1
                self._enemies.append(e)

    def _game_step(
        self, action_name: str
    ) -> tuple[float, bool, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {}
        self._step_counter += 1

        if action_name == "LEFT" and self._player_x > 1:
            self._player_x -= 1
        elif (
            action_name == "RIGHT"
            and self._player_x < self._WIDTH - 2
        ):
            self._player_x += 1
        elif action_name == "FIRE" and len(self._bullets) < 2:
            b = self._add_entity(
                "bullet", "!", self._player_x,
                self._player_y - 1, dy=-1,
            )
            self._bullets.append(b)

        # Move bullets
        for b in self._bullets:
            if not b.alive:
                continue
            b.y += b.dy
            if b.y <= 0:
                b.alive = False

        # Bullet-enemy collisions
        for b in self._bullets:
            if not b.alive:
                continue
            for e in self._enemies:
                if e.alive and e.x == b.x and e.y == b.y:
                    e.alive = False
                    b.alive = False
                    pts = 25 if e.etype == "leader" else 10
                    self._on_point_scored(pts)
                    reward += pts
                    self._message = f"Enemy hit! +{pts}"
                    break
        self._bullets = [b for b in self._bullets if b.alive]

        # Move enemies
        if self._step_counter % 3 == 0:
            for e in self._enemies:
                if not e.alive:
                    continue
                d = e.data.get("dir", 1)
                e.x += d
                if e.x <= 1 or e.x >= self._WIDTH - 2:
                    e.data["dir"] = -d
                    e.y += 1

        # Enemy fire
        if self._step_counter % 5 == 0:
            alive = [e for e in self._enemies if e.alive]
            if alive and len(self._enemy_bullets) < 3:
                s = alive[int(self.rng.integers(len(alive)))]
                eb = self._add_entity(
                    "enemy_bullet", "v", s.x, s.y + 1, dy=1,
                )
                self._enemy_bullets.append(eb)

        # Move enemy bullets
        for eb in self._enemy_bullets:
            if not eb.alive:
                continue
            eb.y += eb.dy
            if eb.y >= self._HEIGHT - 1:
                eb.alive = False
            elif (
                eb.x == self._player_x
                and eb.y == self._player_y
            ):
                eb.alive = False
                self._on_life_lost()
                self._message = "Hit! Lost a life."
                self._player_x = self._WIDTH // 2
        self._enemy_bullets = [
            eb for eb in self._enemy_bullets if eb.alive
        ]

        # Enemy reaches bottom
        for e in self._enemies:
            if e.alive and e.y >= self._PLAYER_Y:
                e.alive = False
                self._on_life_lost()
                self._message = "Enemy reached you!"
        self._enemies = [e for e in self._enemies if e.alive]

        # Spawn new formations periodically
        self._spawn_timer += 1
        if not self._enemies or self._spawn_timer >= 30:
            if not self._enemies:
                self._level += 1
                self._message = "Wave cleared!"
            self._spawn_timer = 0
            self._spawn_formation()

        self._redraw()
        return reward, self._game_over, info

    def _redraw(self) -> None:
        for y in range(1, self._HEIGHT - 1):
            for x in range(1, self._WIDTH - 1):
                self._set_cell(x, y, " ")
        for e in self._enemies:
            if e.alive:
                self._set_cell(e.x, e.y, e.char)
        for b in self._bullets:
            if b.alive:
                self._set_cell(b.x, b.y, "!")
        for eb in self._enemy_bullets:
            if eb.alive:
                self._set_cell(eb.x, eb.y, "v")
        # Draw turret base
        if 1 < self._player_x < self._WIDTH - 2:
            self._set_cell(
                self._player_x - 1, self._PLAYER_Y + 0, "["
            )
            self._set_cell(
                self._player_x + 1, self._PLAYER_Y + 0, "]"
            )

    def _advance_entities(self) -> None:
        self._entities = [e for e in self._entities if e.alive]

    def _symbol_meaning(self, ch: str) -> str:
        return {
            "-": "wall", "|": "wall",
            "A": "formation leader (25pts)",
            "a": "enemy (10pts)",
            "!": "your bullet", "v": "enemy bullet",
            "[": "turret base", "]": "turret base",
            " ": "empty",
        }.get(ch, ch)

    def _task_description(self) -> str:
        return (
            "Defend against descending enemy formations. "
            "Shoot leaders for bonus points. "
            "New waves spawn continuously."
        )
