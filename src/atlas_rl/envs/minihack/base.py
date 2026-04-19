"""MiniHackBase: shared base class for all MiniHack environments.

Handles grid state, player, 8-directional movement, combat, monsters,
traps, dark rooms, and NetHack-style ASCII rendering.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from atlas_rl.core.action import ActionSpec
from atlas_rl.core.ascii_primitives import build_legend, grid_to_string, make_empty_grid
from atlas_rl.core.base_env import BaseAsciiEnv
from atlas_rl.core.observation import GridObservation
from atlas_rl.envs.minihack.creatures import Creature, CreatureType
from atlas_rl.envs.minihack.items import Item

# Shared MiniHack action spec (22 actions)
MINIHACK_ACTION_SPEC = ActionSpec(
    names=(
        "MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W",
        "MOVE_NE", "MOVE_NW", "MOVE_SE", "MOVE_SW",
        "WAIT", "SEARCH", "LOOK",
        "PICKUP", "DROP", "EAT", "READ", "QUAFF",
        "WIELD", "ZAP", "PRAY",
        "APPLY", "INVENTORY", "ESCAPE",
    ),
    descriptions=(
        "move one cell north (up)",
        "move one cell south (down)",
        "move one cell east (right)",
        "move one cell west (left)",
        "move one cell northeast (up-right)",
        "move one cell northwest (up-left)",
        "move one cell southeast (down-right)",
        "move one cell southwest (down-left)",
        "wait one turn (no-op)",
        "search the area around you",
        "look around",
        "pick up an item at your feet",
        "drop an item",
        "eat food from inventory",
        "read a scroll from inventory",
        "drink a potion from inventory",
        "wield a weapon from inventory",
        "zap a wand (at nearest target)",
        "pray for divine help",
        "apply/use an item",
        "check your inventory",
        "escape/cancel",
    ),
)

# Direction vectors for 8-directional movement
MOVE_VECTORS: dict[str, tuple[int, int]] = {
    "MOVE_N": (0, -1),
    "MOVE_S": (0, 1),
    "MOVE_E": (1, 0),
    "MOVE_W": (-1, 0),
    "MOVE_NE": (1, -1),
    "MOVE_NW": (-1, -1),
    "MOVE_SE": (1, 1),
    "MOVE_SW": (-1, 1),
}


class MiniHackBase(BaseAsciiEnv):
    """Abstract base for all MiniHack environments.

    Subclasses MUST implement:
      - ``env_id()`` -> str
      - ``_generate_level(seed)`` -> None  (must call ``_init_grid``,
        ``_place_player``, and optionally ``_place_stairs``, etc.)
    """

    action_spec = MINIHACK_ACTION_SPEC
    noop_action_name: str = "WAIT"

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)
        self._grid: list[list[str]] = []
        self._grid_w: int = 0
        self._grid_h: int = 0
        self._player_pos: tuple[int, int] = (0, 0)
        self._player_hp: int = 12
        self._player_max_hp: int = 12
        self._goal_pos: tuple[int, int] | None = None
        self._creatures: list[Creature] = []
        self._message: str = ""
        self._dark: bool = False
        self._vision_radius: int = 1  # for dark rooms
        self._inventory: list[Item] = []
        self._floor_items: dict[tuple[int, int], list[Item]] = {}
        self._wielding: Item | None = None
        self._hunger: int = 100  # 100 = full, 0 = starving

    # ------------------------------------------------------------------
    # Grid setup helpers (called from _generate_level)
    # ------------------------------------------------------------------

    def _init_grid(self, width: int, height: int) -> None:
        """Create grid with wall border. Uses ``-`` for top/bottom, ``|`` for sides."""
        self._grid_w = width
        self._grid_h = height
        self._grid = [["." for _ in range(width)] for _ in range(height)]
        for x in range(width):
            self._grid[0][x] = "-"
            self._grid[height - 1][x] = "-"
        for y in range(1, height - 1):
            self._grid[y][0] = "|"
            self._grid[y][width - 1] = "|"

    def _place_player(self, x: int, y: int) -> None:
        self._player_pos = (x, y)

    def _place_stairs(self, x: int, y: int) -> None:
        self._grid[y][x] = ">"
        self._goal_pos = (x, y)

    def _place_wall(self, x: int, y: int) -> None:
        self._grid[y][x] = "#"

    def _place_trap(self, x: int, y: int) -> None:
        self._grid[y][x] = "^"

    def _place_lava(self, x: int, y: int) -> None:
        self._grid[y][x] = "}"

    def _place_water(self, x: int, y: int) -> None:
        self._grid[y][x] = "~"

    def _place_door(self, x: int, y: int) -> None:
        self._grid[y][x] = "+"

    def _place_item(self, x: int, y: int, item: Item) -> None:
        key = (x, y)
        if key not in self._floor_items:
            self._floor_items[key] = []
        self._floor_items[key].append(item)

    def _spawn_creature(self, ctype: CreatureType, x: int, y: int) -> None:
        self._creatures.append(Creature.spawn(ctype, x, y))

    def _terrain_at(self, x: int, y: int) -> str:
        if 0 <= x < self._grid_w and 0 <= y < self._grid_h:
            return self._grid[y][x]
        return "#"  # out of bounds = wall

    def _is_walkable(self, x: int, y: int) -> bool:
        t = self._terrain_at(x, y)
        return t in (".", ">", "^", "+", "}", "~")

    def _creature_at(self, x: int, y: int) -> Creature | None:
        for c in self._creatures:
            if c.x == x and c.y == y and c.hp > 0:
                return c
        return None

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._player_hp = self._player_max_hp
        self._creatures = []
        self._message = ""
        self._goal_pos = None
        self._inventory = []
        self._floor_items = {}
        self._wielding = None
        self._hunger = 100
        self._generate_level(seed)
        return self._render_current_observation()

    @abstractmethod
    def _generate_level(self, seed: int) -> None:
        """Subclass must call ``_init_grid``, ``_place_player``, ``_place_stairs``, etc."""
        ...

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        self._message = ""
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        if name in MOVE_VECTORS:
            dx, dy = MOVE_VECTORS[name]
            nx, ny = self._player_pos[0] + dx, self._player_pos[1] + dy

            # Check for monster at target -- melee combat
            monster = self._creature_at(nx, ny)
            if monster is not None:
                # Attack monster
                dmg = max(1, int(self.rng.integers(1, 5)))
                monster.hp -= dmg
                self._message = f"You hit the {monster.ctype.name}!"
                if monster.hp <= 0:
                    self._message += f" The {monster.ctype.name} dies."
                    self._creatures = [c for c in self._creatures if c.hp > 0]
            elif self._is_walkable(nx, ny):
                self._player_pos = (nx, ny)

                # Check terrain effects
                terrain = self._terrain_at(nx, ny)
                if terrain == "^":
                    trap_dmg = int(self.rng.integers(1, 4))
                    self._player_hp -= trap_dmg
                    self._message = f"You fall into a trap! (-{trap_dmg} HP)"
                    self._grid[ny][nx] = "."  # trap sprung
                elif terrain == "}":
                    self._player_hp = 0
                    self._message = "You fall into lava!"
                elif terrain == "~":
                    info["water"] = True
        elif name == "PICKUP":
            px, py = self._player_pos
            items = self._floor_items.get((px, py), [])
            if items:
                item = items.pop(0)
                self._inventory.append(item)
                self._message = f"You pick up the {item.name}."
                if not items:
                    del self._floor_items[(px, py)]
        elif name == "DROP":
            if self._inventory:
                item = self._inventory.pop(0)
                px, py = self._player_pos
                self._place_item(px, py, item)
                self._message = f"You drop the {item.name}."
        elif name == "EAT":
            food = next(
                (i for i in self._inventory if i.item_type == "food"), None
            )
            if food:
                self._inventory.remove(food)
                self._hunger = min(100, self._hunger + 50)
                self._message = f"You eat the {food.name}. Yum!"
        elif name == "READ":
            scroll = next(
                (i for i in self._inventory if i.item_type == "scroll"), None
            )
            if scroll:
                self._inventory.remove(scroll)
                self._message = f"You read the {scroll.name}."
                self._on_read_scroll(scroll)
        elif name == "QUAFF":
            potion = next(
                (i for i in self._inventory if i.item_type == "potion"), None
            )
            if potion:
                self._inventory.remove(potion)
                self._message = f"You drink the {potion.name}."
                self._on_quaff_potion(potion)
        elif name == "WIELD":
            weapon = next(
                (i for i in self._inventory if i.item_type == "weapon"), None
            )
            if weapon:
                self._wielding = weapon
                self._message = f"You wield the {weapon.name}."
        elif name == "ZAP":
            wand = next(
                (i for i in self._inventory if i.item_type == "wand"), None
            )
            if wand:
                self._inventory.remove(wand)
                self._message = f"You zap the {wand.name}!"
                self._on_zap_wand(wand)
        elif name == "PRAY":
            self._message = "You pray to the gods."
            self._on_pray()
        # WAIT, SEARCH, LOOK, APPLY, INVENTORY, ESCAPE: no-op for navigation tasks

        # Check player death
        if self._player_hp <= 0:
            terminated = True
            self._message = (self._message + " You die.").strip()
            info["cause_of_death"] = (
                "combat" if "hit" in self._message.lower() else "hazard"
            )
            return self._render_current_observation(), -1.0, terminated, False, info

        # Monster turns
        self._move_monsters()

        # Check if monster killed player
        if self._player_hp <= 0:
            terminated = True
            self._message = (self._message + " You die.").strip()
            info["cause_of_death"] = "monster"
            return self._render_current_observation(), -1.0, terminated, False, info

        # Check goal
        if self._goal_pos and self._player_pos == self._goal_pos:
            terminated = True
            reward = 1.0
            self._message = "You reach the stairs. You descend."
            info["goal_reached"] = True

        info["player_pos"] = self._player_pos
        info["hp"] = self._player_hp
        return self._render_current_observation(), reward, terminated, False, info

    def _move_monsters(self) -> None:
        """Hostile monsters move toward player."""
        px, py = self._player_pos
        for c in self._creatures:
            if c.hp <= 0 or c.ctype.ai != "hostile":
                continue
            # Simple chase AI: move toward player
            dx = 0 if c.x == px else (1 if c.x < px else -1)
            dy = 0 if c.y == py else (1 if c.y < py else -1)
            nx, ny = c.x + dx, c.y + dy

            # Check if new pos is player -- attack
            if (nx, ny) == (px, py):
                dmg = max(1, c.ctype.damage)
                self._player_hp -= dmg
                self._message += f" The {c.ctype.name} hits you! (-{dmg} HP)"
                continue

            # Move if walkable and no other creature there
            if self._is_walkable(nx, ny) and self._creature_at(nx, ny) is None:
                c.x, c.y = nx, ny

    # ------------------------------------------------------------------
    # Item-action hooks (override in subclasses for effects)
    # ------------------------------------------------------------------

    def _on_read_scroll(self, scroll: Item) -> None:
        """Override for scroll effects."""

    def _on_quaff_potion(self, potion: Item) -> None:
        """Override for potion effects."""

    def _on_zap_wand(self, wand: Item) -> None:
        """Override for wand effects."""

    def _on_pray(self) -> None:
        """Override for prayer effects."""

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current_observation(self) -> GridObservation:
        render = make_empty_grid(self._grid_w, self._grid_h, fill=" ")
        symbols: dict[str, str] = {}

        px, py = self._player_pos

        for y in range(self._grid_h):
            for x in range(self._grid_w):
                # Dark room visibility check
                if self._dark and (
                    abs(x - px) > self._vision_radius
                    or abs(y - py) > self._vision_radius
                ):
                    continue  # leave as space (unseen)

                ch = self._grid[y][x]
                render[y][x] = ch
                if ch == ".":
                    symbols["."] = "floor"
                elif ch == "-":
                    symbols["-"] = "wall"
                elif ch == "|":
                    symbols["|"] = "wall"
                elif ch == "#":
                    symbols["#"] = "wall"
                elif ch == ">":
                    symbols[">"] = "stairs down (goal)"
                elif ch == "^":
                    symbols["^"] = "trap"
                elif ch == "}":
                    symbols["}"] = "lava"
                elif ch == "~":
                    symbols["~"] = "water"
                elif ch == "+":
                    symbols["+"] = "door"

        # Floor items
        for (ix, iy), items in self._floor_items.items():
            if items and (
                not self._dark
                or (
                    abs(ix - px) <= self._vision_radius
                    and abs(iy - py) <= self._vision_radius
                )
            ):
                ch = items[0].char
                render[iy][ix] = ch
                symbols[ch] = items[0].legend_name()

        # Creatures
        for c in self._creatures:
            if c.hp <= 0:
                continue
            if self._dark and (
                abs(c.x - px) > self._vision_radius
                or abs(c.y - py) > self._vision_radius
            ):
                continue
            render[c.y][c.x] = c.ctype.char
            symbols[c.ctype.char] = c.ctype.name

        # Player on top
        render[py][px] = "@"
        symbols["@"] = "you"

        legend = build_legend(symbols)

        dark_note = "  Vision: limited (dark room)" if self._dark else ""
        inv_note = f"  Inv: {len(self._inventory)}" if self._inventory else ""
        hud = (
            f"Dlvl: 1    HP: {self._player_hp}/{self._player_max_hp}    "
            f"Turn: {self._turn}    Pos: ({px},{py}){dark_note}{inv_note}"
        )

        return GridObservation(
            grid=grid_to_string(render),
            legend=legend,
            hud=hud,
            message=self._message,
        )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _task_description(self) -> str:
        return "Navigate to the stairs down (>) to descend. Reward: +1 on reaching stairs."

    def system_prompt(self) -> str:
        return (
            f"You are playing {self.env_id()}.\n\n"
            f"TASK\n{self._task_description()}\n\n"
            "GRID\n"
            "NetHack-style ASCII dungeon. @ is you. > is stairs down (goal). "
            ". is floor. | and - are walls. + is a door.\n\n"
            "MOVEMENT\n"
            "8 directional movement: N, S, E, W, NE, NW, SE, SW. "
            "Moving into a wall does nothing. Moving into a monster attacks it.\n\n"
            + self.action_spec.render_for_prompt()
        )
