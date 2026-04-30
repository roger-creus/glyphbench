"""Craftax Full environment (80 achievements, phase β).

Extends Craftax Classic with multi-floor dungeons, magic, bosses,
armor, enchantments, potions, and new mob types.

Gym ID: glyphbench/craftax-v0
"""

from __future__ import annotations

import random
from typing import Any, TypedDict

import numpy as np

from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation
from glyphbench.envs.craftax.base import (
    ALL_FULL_ACHIEVEMENTS,
    CRAFTAX_FULL_ACTION_SPEC,
    FULL_VIEW_HEIGHT,
    FULL_VIEW_WIDTH,
    TILE_AGENT,
    TILE_ARROW,
    TILE_ARROW2,
    TILE_BAT,
    TILE_BOSS,
    TILE_BOSS_DOOR,
    TILE_CHEST,
    TILE_COAL,
    TILE_COW,
    TILE_DAGGER,
    TILE_DIAMOND,
    TILE_DUNGEON_FLOOR,
    TILE_DUNGEON_WALL,
    TILE_FIREBALL,
    TILE_FIREBALL2,
    TILE_FOUNTAIN,
    TILE_FURNACE,
    TILE_GRASS,
    TILE_ICEBALL,
    TILE_ICEBALL2,
    TILE_IRON,
    TILE_LAVA,
    TILE_PLACED_STONE,
    TILE_PLANT,
    TILE_RIPE_PLANT,
    TILE_SAND,
    TILE_SAPLING,
    TILE_SKELETON_ARCHER,
    TILE_SLIMEBALL,
    TILE_KOBOLD,
    TILE_RUBY,
    TILE_SAPPHIRE,
    TILE_ENCHANT_FIRE,
    TILE_STAIRS_DOWN,
    TILE_STAIRS_UP,
    TILE_STONE,
    TILE_TABLE,
    TILE_TORCH,
    TILE_TREE,
    TILE_WATER,
    TILE_ZOMBIE,
)

# ----------------------------------------------------------------
# Upstream achievement names (from Craftax-main constants.py Achievement enum)
# Used by phase-β tasks to fire upstream-named achievements.
# ----------------------------------------------------------------
UPSTREAM_ACHIEVEMENT_NAMES: tuple[str, ...] = (
    "COLLECT_WOOD",
    "PLACE_TABLE",
    "EAT_COW",
    "COLLECT_SAPLING",
    "COLLECT_DRINK",
    "MAKE_WOOD_PICKAXE",
    "MAKE_WOOD_SWORD",
    "PLACE_PLANT",
    "DEFEAT_ZOMBIE",
    "COLLECT_STONE",
    "PLACE_STONE",
    "EAT_PLANT",
    "DEFEAT_SKELETON",
    "MAKE_STONE_PICKAXE",
    "MAKE_STONE_SWORD",
    "WAKE_UP",
    "PLACE_FURNACE",
    "COLLECT_COAL",
    "COLLECT_IRON",
    "COLLECT_DIAMOND",
    "MAKE_IRON_PICKAXE",
    "MAKE_IRON_SWORD",
    "MAKE_ARROW",
    "MAKE_TORCH",
    "PLACE_TORCH",
    "COLLECT_SAPPHIRE",
    "COLLECT_RUBY",
    "MAKE_DIAMOND_PICKAXE",
    "MAKE_DIAMOND_SWORD",
    "MAKE_IRON_ARMOUR",
    "MAKE_DIAMOND_ARMOUR",
    "ENTER_GNOMISH_MINES",
    "ENTER_DUNGEON",
    "ENTER_SEWERS",
    "ENTER_VAULT",
    "ENTER_TROLL_MINES",
    "ENTER_FIRE_REALM",
    "ENTER_ICE_REALM",
    "ENTER_GRAVEYARD",
    "DEFEAT_GNOME_WARRIOR",
    "DEFEAT_GNOME_ARCHER",
    "DEFEAT_ORC_SOLIDER",
    "DEFEAT_ORC_MAGE",
    "DEFEAT_LIZARD",
    "DEFEAT_KOBOLD",
    "DEFEAT_KNIGHT",
    "DEFEAT_ARCHER",
    "DEFEAT_TROLL",
    "DEFEAT_DEEP_THING",
    "DEFEAT_PIGMAN",
    "DEFEAT_FIRE_ELEMENTAL",
    "DEFEAT_FROST_TROLL",
    "DEFEAT_ICE_ELEMENTAL",
    "DAMAGE_NECROMANCER",
    "DEFEAT_NECROMANCER",
    "EAT_BAT",
    "EAT_SNAIL",
    "FIND_BOW",
    "FIRE_BOW",
    "LEARN_FIREBALL",
    "CAST_FIREBALL",
    "LEARN_ICEBALL",
    "CAST_ICEBALL",
    "OPEN_CHEST",
    "DRINK_POTION",
    "ENCHANT_SWORD",
    "ENCHANT_ARMOUR",
)

# ----------------------------------------------------------------
# Walkable / interactable tile sets
# ----------------------------------------------------------------
SURFACE_WALKABLE = frozenset({
    TILE_GRASS, TILE_SAND, TILE_TABLE, TILE_FURNACE,
    TILE_PLACED_STONE, TILE_SAPLING, TILE_RIPE_PLANT,
    TILE_STAIRS_DOWN, TILE_TORCH,
    # Phase β T07β: lava is now walkable but deals 2 dmg/tick.
    TILE_LAVA,
})

DUNGEON_WALKABLE = frozenset({
    TILE_DUNGEON_FLOOR, TILE_TABLE, TILE_FURNACE,
    TILE_PLACED_STONE, TILE_STAIRS_DOWN, TILE_STAIRS_UP,
    TILE_TORCH, TILE_BOSS_DOOR,
    # Phase β T07β: lava is walkable in dungeons too.
    TILE_LAVA,
})

INTERACTABLE_TILES: dict[str, str] = {
    TILE_TREE: "wood",
    TILE_STONE: "stone",
    TILE_COAL: "coal",
    TILE_IRON: "iron",
    TILE_DIAMOND: "diamond",
    # Phase β ore gems (T05β/T06β)
    TILE_SAPPHIRE: "sapphire",
    TILE_RUBY: "ruby",
}

PICKAXE_REQUIRED: dict[str, int] = {
    TILE_STONE: 0,
    TILE_COAL: 1,
    TILE_IRON: 1,
    TILE_DIAMOND: 2,
    # Gem ores require iron pickaxe (tier 2 = iron_pickaxe or better)
    TILE_SAPPHIRE: 2,
    TILE_RUBY: 2,
}

_PICKAXE_TIERS = (
    "wood_pickaxe", "stone_pickaxe",
    "iron_pickaxe", "diamond_pickaxe",
)

# ----------------------------------------------------------------
# Timing constants
# ----------------------------------------------------------------
_DAY_LENGTH = 200
_NIGHT_LENGTH = 100
_CYCLE_LENGTH = _DAY_LENGTH + _NIGHT_LENGTH

# Night mob spawn base chance (phase β T17β).
# At full darkness (light=0.0): effective_chance = _NIGHT_SPAWN_BASE_CHANCE * (1-0)² = 1.0
# At night ambient (light=0.3): effective_chance ≈ 0.49
# At full light (light=1.0):  effective_chance = 0.0 (no spawn)
_NIGHT_SPAWN_BASE_CHANCE = 1.0

_FOOD_DRAIN_INTERVAL = 50
_WATER_DRAIN_INTERVAL = 40
_ENERGY_DRAIN_INTERVAL = 100

_MAX_FOOD = 9
_MAX_WATER = 9
_MAX_ENERGY = 9
_MAX_MANA = 10
_MANA_REGEN_INTERVAL = 20

_PLANT_RIPEN_STEPS = 20
_SAPLING_DROP_CHANCE = 0.3

# Directional player characters
_DIR_CHARS: dict[tuple[int, int], str] = {
    (1, 0): "\u2192", (-1, 0): "\u2190", (0, -1): "\u2191", (0, 1): "\u2193",  # →←↑↓
}
_DIR_NAMES: dict[tuple[int, int], str] = {
    (1, 0): "right", (-1, 0): "left",
    (0, -1): "up", (0, 1): "down",
}

# Spell names ordered by learn index
_SPELL_NAMES = ("fireball", "iceball", "heal")

# Solid tiles that block projectiles (they cannot pass through walls / trees /
# stone / placed stone / closed boss doors).
_SOLID_TILES: frozenset[str] = frozenset({
    TILE_DUNGEON_WALL, TILE_TREE, TILE_STONE, TILE_PLACED_STONE,
    TILE_BOSS_DOOR,
    # Crafting structures and ores also block projectiles.
    TILE_TABLE, TILE_FURNACE, TILE_COAL, TILE_IRON, TILE_DIAMOND,
    # Phase β gem ores block projectiles too.
    TILE_SAPPHIRE, TILE_RUBY,
    # Plants block projectiles too (upstream constants.py:370-371).
    TILE_PLANT, TILE_RIPE_PLANT,
})

# ----------------------------------------------------------------
# World sizes
# ----------------------------------------------------------------
_SURFACE_SIZE = 64
_DUNGEON_SIZE = 32
_NUM_DUNGEON_FLOORS = 5

# ----------------------------------------------------------------
# Mob definitions
# ----------------------------------------------------------------
_MOB_STATS: dict[str, dict[str, int]] = {
    # Upstream-faithful roster for the full env (T_FOLLOWUP_A / T04β).
    # "zombie" = melee floor-0 mob; "skeleton" = ranged floor-0 mob (upstream).
    # "kobold" replaces legacy "spider" (upstream ranged mob, throws daggers).
    "zombie": {"hp": 3, "damage": 1},
    "cow": {"hp": 3, "damage": 0},
    "skeleton": {"hp": 5, "damage": 3},   # upstream ranged; was skeleton_archer
    "kobold": {"hp": 4, "damage": 2},      # upstream ranged; was spider
    "bat": {"hp": 2, "damage": 1},
}

_MOB_TILES: dict[str, str] = {
    "zombie": TILE_ZOMBIE,
    "cow": TILE_COW,
    "skeleton": TILE_SKELETON_ARCHER,  # glyph "a" (upstream archer convention)
    "kobold": TILE_KOBOLD,             # glyph "q"
    "bat": TILE_BAT,
}

# Boss definitions per floor (1-indexed)
_BOSS_DEFS: dict[int, dict[str, Any]] = {
    1: {"name": "knight", "hp": 10, "damage": 3,
        "ach": "defeat_knight", "ranged": False},
    2: {"name": "archer_boss", "hp": 12, "damage": 4,
        "ach": "defeat_archer_boss", "ranged": True},
    3: {"name": "mage", "hp": 8, "damage": 5,
        "ach": "defeat_mage", "ranged": True},
    4: {"name": "dragon", "hp": 20, "damage": 6,
        "ach": "defeat_dragon", "ranged": False},
    5: {"name": "lich", "hp": 15, "damage": 7,
        "ach": "defeat_lich", "ranged": True},
}

# Weapon damage bonuses
_WEAPON_BONUS: dict[str, int] = {
    "wood_sword": 1,
    "stone_sword": 2,
    "iron_sword": 3,
    "diamond_sword": 4,
}

# Armor defense values
_ARMOR_DEFENSE: dict[str, int] = {
    "wood_armor": 1,
    "stone_armor": 2,
    "iron_armor": 3,
    "diamond_armor": 4,
}


class Mob(TypedDict):
    """Mob state dictionary."""

    type: str
    x: int
    y: int
    hp: int
    max_hp: int
    is_boss: bool
    floor: int
    attack_cooldown: int  # ticks remaining before this mob can attack again


class CraftaxFullEnv(BaseGlyphEnv):
    """Craftax Full: survival crafting with dungeons and magic.

    80 achievements spanning resource gathering, crafting, combat,
    dungeon exploration, magic, bosses, and survival milestones.

    Surface: 64x64, Dungeons: 32x32 per floor, 5 floors.
    Visible window: 11x9 centered on agent.
    Reward: +1 per first-time achievement unlock.
    """

    action_spec = CRAFTAX_FULL_ACTION_SPEC
    noop_action_name = "NOOP"

    _ALL_ACHIEVEMENTS = ALL_FULL_ACHIEVEMENTS

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        # World state
        self._floors: dict[int, list[list[str]]] = {}
        self._current_floor: int = 0  # 0 = surface
        self._agent_x: int = 0
        self._agent_y: int = 0
        self._facing: tuple[int, int] = (1, 0)
        # Inventory and equipment
        self._inventory: dict[str, int] = {}
        self._weapon_enchanted: bool = False
        self._armor_enchanted: bool = False
        # Achievements
        self._achievements_unlocked: set[str] = set()
        # Phase-β parallel bitmap keyed by upstream Achievement enum names.
        self._achievements_phase_beta: dict[str, bool] = {
            n: False for n in UPSTREAM_ACHIEVEMENT_NAMES
        }
        self._message: str = ""
        # Vitals
        self._hp: int = 9
        self._max_hp: int = 9
        self._food: int = _MAX_FOOD
        self._water: int = _MAX_WATER
        self._energy: int = _MAX_ENERGY
        self._mana: int = _MAX_MANA
        # Spells learned (per-spell dict; T10β)
        self._learned_spells: dict[str, bool] = {"fireball": False, "iceball": False}
        # Day/night
        self._day_counter: int = 0
        self._day_night: str = "day"
        self._night_count: int = 0
        # Mobs (all floors)
        self._mobs: list[Mob] = []
        # Projectile entities (phase α).
        from glyphbench.envs.craftax.mechanics.projectiles import ProjectileEntity
        self._player_projectiles: list[ProjectileEntity] = []
        self._mob_projectiles: list[ProjectileEntity] = []
        self._is_sleeping: bool = False
        self._is_resting: bool = False
        self._pending_step_reward: float = 0.0
        # Plants
        self._plants: dict[tuple[int, int], int] = {}
        # Potions — phase β: per-color dict + hidden per-episode mapping
        self._potions: list[str] = []  # legacy field kept for forward-compat; superseded by potions dict in inventory
        self._potion_mapping: tuple[int, ...] = (0, 1, 2, 3, 4, 5)  # will be set in _reset
        # Active effects
        self._speed_turns: int = 0
        # Milestone counters
        self._total_kills: int = 0
        self._total_crafts: int = 0
        self._total_blocks_placed: int = 0
        self._total_plants_eaten: int = 0
        self._total_water_drunk: int = 0
        # Torch visibility in dungeons
        self._torches: dict[int, set[tuple[int, int]]] = {}
        # Per-tile lightmap (phase β T15β): floor -> np.ndarray[h,w] in [0,1]
        self._lightmap: dict[int, np.ndarray] = {}
        # Stairs locations  floor -> (x, y)
        self._stairs_down_pos: dict[int, tuple[int, int]] = {}
        self._stairs_up_pos: dict[int, tuple[int, int]] = {}
        # Boss alive tracking
        self._bosses_alive: dict[int, bool] = {}
        # Chest tracking (T12β): per-floor set of opened (x, y) coords.
        self._chests_opened: dict[int, set[tuple[int, int]]] = {}
        # First-chest gating (T14β): per-floor bool for first-chest bonus.
        self._first_chest_opened: dict[int, bool] = {}

    # ---------------------------------------------------------------
    # Identity
    # ---------------------------------------------------------------

    def env_id(self) -> str:
        return "glyphbench/craftax-v0"

    def system_prompt(self) -> str:
        ach_list = ", ".join(self._ALL_ACHIEVEMENTS)
        return (
            "You are playing Craftax Full.\n\n"
            "TASK\n"
            "Gather resources, craft tools, fight mobs, explore "
            "dungeons, learn magic, and survive. Each new "
            "achievement gives +1 reward.\n"
            f"Achievements ({len(self._ALL_ACHIEVEMENTS)}): "
            f"{ach_list}.\n\n"
            "WORLD\n"
            "Surface: 64x64 grid. Dungeons: 5 floors (32x32). "
            "11x9 view centered on you.\n"
            "Biomes: grass(.), tree(T), stone(S), coal(C), "
            "iron(I), diamond(D), water(~), lava(L), sand(s).\n"
            "Dungeon: wall(#), floor(_), stairs down(>), "
            "stairs up(<), torch(!), boss door(B).\n"
            "Mobs: zombie(z), cow(c), "
            "skeleton(a), kobold(q), bat(b), boss(W).\n\n"
            "SURVIVAL\n"
            "- Food drains 1/50 steps. 0 food: -1 HP/step.\n"
            "- Water drains 1/40 steps. 0 water: -1 HP/step.\n"
            "- Energy drains 1/100 steps. 0: 50% move fail.\n"
            "- Mana: max 10, regen 1/20 steps. Used for spells.\n\n"
            "COMBAT\n"
            "DO facing mob attacks. Dmg = 1 + weapon bonus "
            "(+enchant). Armor reduces incoming damage.\n"
            "WEAPON DAMAGE: wood sword +1, stone sword +2, "
            "iron sword +3, diamond sword +4.\n"
            "ARMOR DEFENSE: wood armor 1, stone armor 2, "
            "iron armor 3, diamond armor 4.\n"
            "Enchant adds +2 weapon dmg or +1 armor def.\n\n"
            "CRAFTING (at adjacent table)\n"
            "MAKE_ARROW: 1 wood + 1 stone -> 2 arrows (table required).\n"
            "MAKE_TORCH: 1 wood + 1 coal -> 4 torches (table required).\n"
            "Note: bows are not craftable in phase α; they drop from chests (phase β).\n\n"
            "RANGED\n"
            "SHOOT_ARROW: fires 1 arrow forward (requires bow + arrow).\n\n"
            "MAGIC\n"
            "CAST_FIREBALL / CAST_ICEBALL (2 mana each): spawn a fireball / iceball projectile one tile in front of you; travels 1 tile/turn until it hits a target or wall.\n\n"
            "PROJECTILES (mid-flight glyphs)\n"
            "↗/↘=arrow, †=dagger, ●/◉=fireball, ○/◎=iceball, ◐=slimeball.\n\n"
            "DUNGEONS\n"
            "DESCEND on > goes deeper. ASCEND on < goes up. "
            "Dungeons are dark; PLACE_TORCH (consumes 1 crafted torch) for light. "
            "Each floor has a boss (W behind B door). "
            "Floor 4 (Vaults) contains a fire enchantment table (Ⓔ); "
            "enchantment-table interaction is unlocked in phase γ.\n\n"
            + self.action_spec.render_for_prompt()
        )

    # ---------------------------------------------------------------
    # Floor helpers
    # ---------------------------------------------------------------

    def _floor_size(self) -> int:
        if self._current_floor == 0:
            return _SURFACE_SIZE
        return _DUNGEON_SIZE

    def _tile_at(self, x: int, y: int) -> str:
        return self._floors[self._current_floor][y][x]

    def _is_in_bounds(self, x: int, y: int) -> bool:
        size = self._floor_size()
        return 0 <= x < size and 0 <= y < size

    def _current_grid(self) -> list[list[str]]:
        return self._floors[self._current_floor]

    def _walkable_set(self) -> frozenset[str]:
        if self._current_floor == 0:
            return SURFACE_WALKABLE
        return DUNGEON_WALKABLE

    # ---------------------------------------------------------------
    # World generation
    # ---------------------------------------------------------------

    def _generate_surface(self) -> None:
        size = _SURFACE_SIZE
        grid = [
            [TILE_GRASS for _ in range(size)]
            for _ in range(size)
        ]

        # Forests
        for _ in range(int(self.rng.integers(8, 15))):
            cx = int(self.rng.integers(5, size - 5))
            cy = int(self.rng.integers(5, size - 5))
            r = int(self.rng.integers(3, 7))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        x, y = cx + dx, cy + dy
                        if (
                            0 <= x < size
                            and 0 <= y < size
                            and self.rng.random() < 0.7
                        ):
                            grid[y][x] = TILE_TREE

        # Stone deposits
        for _ in range(int(self.rng.integers(5, 10))):
            cx = int(self.rng.integers(5, size - 5))
            cy = int(self.rng.integers(5, size - 5))
            r = int(self.rng.integers(2, 4))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        x, y = cx + dx, cy + dy
                        if (
                            0 <= x < size
                            and 0 <= y < size
                            and self.rng.random() < 0.6
                        ):
                            grid[y][x] = TILE_STONE

        # Coal
        for _ in range(int(self.rng.integers(3, 6))):
            cx = int(self.rng.integers(10, size - 10))
            cy = int(self.rng.integers(10, size - 10))
            r = int(self.rng.integers(1, 3))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        x, y = cx + dx, cy + dy
                        if (
                            0 <= x < size
                            and 0 <= y < size
                            and self.rng.random() < 0.5
                        ):
                            grid[y][x] = TILE_COAL

        # Iron
        for _ in range(int(self.rng.integers(3, 5))):
            cx = int(self.rng.integers(10, size - 10))
            cy = int(self.rng.integers(10, size - 10))
            r = int(self.rng.integers(1, 3))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        x, y = cx + dx, cy + dy
                        if (
                            0 <= x < size
                            and 0 <= y < size
                            and self.rng.random() < 0.4
                        ):
                            grid[y][x] = TILE_IRON

        # Water
        for _ in range(int(self.rng.integers(2, 5))):
            cx = int(self.rng.integers(10, size - 10))
            cy = int(self.rng.integers(10, size - 10))
            r = int(self.rng.integers(2, 5))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        x, y = cx + dx, cy + dy
                        if (
                            0 <= x < size
                            and 0 <= y < size
                            and self.rng.random() < 0.8
                        ):
                            grid[y][x] = TILE_WATER

        # Sand
        for _ in range(int(self.rng.integers(2, 5))):
            cx = int(self.rng.integers(8, size - 8))
            cy = int(self.rng.integers(8, size - 8))
            r = int(self.rng.integers(2, 4))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        x, y = cx + dx, cy + dy
                        if (
                            0 <= x < size
                            and 0 <= y < size
                            and grid[y][x] == TILE_GRASS
                            and self.rng.random() < 0.6
                        ):
                            grid[y][x] = TILE_SAND

        # Lava
        for _ in range(int(self.rng.integers(1, 3))):
            cx = int(self.rng.integers(15, size - 15))
            cy = int(self.rng.integers(15, size - 15))
            r = int(self.rng.integers(1, 3))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        x, y = cx + dx, cy + dy
                        if (
                            0 <= x < size
                            and 0 <= y < size
                            and grid[y][x] == TILE_GRASS
                            and self.rng.random() < 0.7
                        ):
                            grid[y][x] = TILE_LAVA

        # Diamond (rare single tiles)
        for _ in range(int(self.rng.integers(1, 3))):
            x = int(self.rng.integers(15, size - 15))
            y = int(self.rng.integers(15, size - 15))
            grid[y][x] = TILE_DIAMOND

        # Clear starting area
        center = size // 2
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                x, y = center + dx, center + dy
                if 0 <= x < size and 0 <= y < size:
                    grid[y][x] = TILE_GRASS

        # Dungeon entrance on surface
        entrance_x = center + 5
        entrance_y = center + 5
        if 0 <= entrance_x < size and 0 <= entrance_y < size:
            grid[entrance_y][entrance_x] = TILE_STAIRS_DOWN
            self._stairs_down_pos[0] = (entrance_x, entrance_y)

        self._floors[0] = grid

    def _generate_dungeon_floor(self, floor: int) -> None:
        """Generate a dungeon floor with rooms and corridors.

        Floors 1 and 3 use the phase-β dungeon-room biome generator
        (``mechanics/world_gen.generate_dungeon_floor``) which places 8
        non-overlapping rooms with chests and fountains.

        Floors 2, 4, 5 continue to use the original generator (floor 2 has
        sapphire/ruby ore; floors 4-5 will get full biome treatment in phase γ).
        """
        # ---- Phase-β biome generator for floors 1, 3, and 4 (T18β/T19β/T20β) ----
        if floor in (1, 3, 4):
            self._generate_dungeon_floor_biome(floor)
            return

        # ---- Legacy generator for floors 2, 5 ----
        size = _DUNGEON_SIZE
        grid = [
            [TILE_DUNGEON_WALL for _ in range(size)]
            for _ in range(size)
        ]

        # Generate rooms
        rooms: list[tuple[int, int, int, int]] = []
        for _ in range(int(self.rng.integers(5, 9))):
            rw = int(self.rng.integers(4, 8))
            rh = int(self.rng.integers(4, 8))
            rx = int(self.rng.integers(1, size - rw - 1))
            ry = int(self.rng.integers(1, size - rh - 1))
            # Check overlap
            overlap = False
            for ex_rx, ex_ry, ex_rw, ex_rh in rooms:
                if (
                    rx < ex_rx + ex_rw + 1
                    and rx + rw + 1 > ex_rx
                    and ry < ex_ry + ex_rh + 1
                    and ry + rh + 1 > ex_ry
                ):
                    overlap = True
                    break
            if overlap:
                continue
            rooms.append((rx, ry, rw, rh))
            for dy in range(rh):
                for dx in range(rw):
                    grid[ry + dy][rx + dx] = TILE_DUNGEON_FLOOR

        # Ensure at least 2 rooms
        if len(rooms) < 2:
            rooms = [(2, 2, 6, 6), (20, 20, 6, 6)]
            for rx, ry, rw, rh in rooms:
                for dy in range(rh):
                    for dx in range(rw):
                        if (
                            0 <= ry + dy < size
                            and 0 <= rx + dx < size
                        ):
                            grid[ry + dy][rx + dx] = (
                                TILE_DUNGEON_FLOOR
                            )

        # Connect rooms with corridors
        for i in range(len(rooms) - 1):
            r1 = rooms[i]
            r2 = rooms[i + 1]
            cx1 = r1[0] + r1[2] // 2
            cy1 = r1[1] + r1[3] // 2
            cx2 = r2[0] + r2[2] // 2
            cy2 = r2[1] + r2[3] // 2
            # Horizontal then vertical
            x = cx1
            while x != cx2:
                if 0 <= x < size and 0 <= cy1 < size:
                    grid[cy1][x] = TILE_DUNGEON_FLOOR
                x += 1 if cx2 > cx1 else -1
            if 0 <= cx2 < size and 0 <= cy1 < size:
                grid[cy1][cx2] = TILE_DUNGEON_FLOOR
            y = cy1
            while y != cy2:
                if 0 <= cx2 < size and 0 <= y < size:
                    grid[y][cx2] = TILE_DUNGEON_FLOOR
                y += 1 if cy2 > cy1 else -1
            if 0 <= cx2 < size and 0 <= cy2 < size:
                grid[cy2][cx2] = TILE_DUNGEON_FLOOR

        # Place stairs up in first room
        r0 = rooms[0]
        up_x = r0[0] + 1
        up_y = r0[1] + 1
        grid[up_y][up_x] = TILE_STAIRS_UP
        self._stairs_up_pos[floor] = (up_x, up_y)

        # Place stairs down in last room (except floor 5)
        if floor < _NUM_DUNGEON_FLOORS:
            rl = rooms[-1]
            down_x = rl[0] + rl[2] - 2
            down_y = rl[1] + rl[3] - 2
            if 0 <= down_x < size and 0 <= down_y < size:
                grid[down_y][down_x] = TILE_STAIRS_DOWN
                self._stairs_down_pos[floor] = (
                    down_x, down_y,
                )

        # Place boss in a dedicated room (middle-ish room)
        boss_room_idx = len(rooms) // 2
        br = rooms[boss_room_idx]
        boss_x = br[0] + br[2] // 2
        boss_y = br[1] + br[3] // 2
        # Place boss door at room entrance (just mark it)
        door_x = br[0]
        door_y = br[1] + br[3] // 2
        if (
            0 <= door_x < size
            and 0 <= door_y < size
            and grid[door_y][door_x] != TILE_STAIRS_UP
            and grid[door_y][door_x] != TILE_STAIRS_DOWN
        ):
            grid[door_y][door_x] = TILE_BOSS_DOOR

        # Spawn boss mob
        bdef = _BOSS_DEFS[floor]
        boss_mob: Mob = {
            "type": bdef["name"],
            "x": boss_x,
            "y": boss_y,
            "hp": bdef["hp"],
            "max_hp": bdef["hp"],
            "is_boss": True,
            "floor": floor,
            "attack_cooldown": 0,
        }
        self._mobs.append(boss_mob)
        self._bosses_alive[floor] = True

        # Scatter dungeon-specific resources
        num_res = int(self.rng.integers(3, 6))
        for _ in range(num_res):
            res = self.rng.choice(
                [TILE_STONE, TILE_COAL, TILE_IRON]
            )
            for _att in range(20):
                rx = int(self.rng.integers(1, size - 1))
                ry = int(self.rng.integers(1, size - 1))
                if grid[ry][rx] == TILE_DUNGEON_FLOOR:
                    grid[ry][rx] = res
                    break

        # Phase β (T06β): sapphire and ruby ore placement.
        # Floor 2 (Gnomish Mines): 2.5% each on dungeon-floor tiles.
        # Floors 4-7 deferred to phase γ.
        if floor == 2:
            for ry in range(size):
                for rx in range(size):
                    if grid[ry][rx] == TILE_DUNGEON_FLOOR:
                        roll = self.rng.random()
                        if roll < 0.025:
                            grid[ry][rx] = TILE_SAPPHIRE
                        elif roll < 0.05:
                            grid[ry][rx] = TILE_RUBY

        # Spawn dungeon mobs
        mob_types = ["skeleton", "kobold", "bat"]
        num_mobs = floor + 2
        for _ in range(num_mobs):
            mtype = str(self.rng.choice(mob_types))
            for _att in range(30):
                mx = int(self.rng.integers(1, size - 1))
                my = int(self.rng.integers(1, size - 1))
                if (
                    grid[my][mx] == TILE_DUNGEON_FLOOR
                    and not self._mob_at(mx, my, floor)
                ):
                    stats = _MOB_STATS[mtype]
                    mob: Mob = {
                        "type": mtype,
                        "x": mx,
                        "y": my,
                        "hp": stats["hp"],
                        "max_hp": stats["hp"],
                        "is_boss": False,
                        "floor": floor,
                        "attack_cooldown": 0,
                    }
                    self._mobs.append(mob)
                    break

        self._floors[floor] = grid
        self._torches[floor] = set()

    def _generate_dungeon_floor_biome(self, floor: int) -> None:
        """Generate floors 1, 3, and 4 using the phase-β dungeon-room biome generator.

        Uses ``mechanics.world_gen.generate_dungeon_floor`` to create a grid
        with 8 rooms, L-shaped corridors, 1 chest per room, and ~50% fountain
        probability per room.  Boss, mob spawning, and resource scattering are
        then layered on top of the generated grid.

        Floor 4 (Vaults) additionally receives 1 ``TILE_ENCHANT_FIRE`` tile
        placed in the middle of the second room (deterministic position given
        the seeded generator).  Phase γ will wire the enchantment-table
        interaction semantics; this task only places the tile.
        """
        from glyphbench.envs.craftax.mechanics.world_gen import generate_dungeon_floor

        size = _DUNGEON_SIZE

        (
            grid,
            _chest_positions,
            _fountain_positions,
            stairs_down_pos,
            stairs_up_pos,
            _agent_spawn,
        ) = generate_dungeon_floor(
            self.rng,
            size,
            num_rooms=8,
            with_chests=True,
            with_fountains=True,
        )

        # Wire stair positions into env state.
        self._stairs_up_pos[floor] = stairs_up_pos
        if floor < _NUM_DUNGEON_FLOORS:
            self._stairs_down_pos[floor] = stairs_down_pos

        # Place boss in a random interior floor cell (avoid stair positions).
        skip_positions = {stairs_up_pos, stairs_down_pos}
        boss_x, boss_y = size // 2, size // 2  # fallback
        bdef = _BOSS_DEFS[floor]
        for _att in range(50):
            bx = int(self.rng.integers(size // 4, 3 * size // 4))
            by = int(self.rng.integers(size // 4, 3 * size // 4))
            if (
                grid[by][bx] == TILE_DUNGEON_FLOOR
                and (bx, by) not in skip_positions
            ):
                boss_x, boss_y = bx, by
                break

        # Place boss door adjacent to boss (one cell to the left).
        door_x = boss_x - 1
        door_y = boss_y
        if (
            0 <= door_x < size
            and grid[door_y][door_x] == TILE_DUNGEON_FLOOR
            and (door_x, door_y) not in skip_positions
        ):
            grid[door_y][door_x] = TILE_BOSS_DOOR

        boss_mob: Mob = {
            "type": bdef["name"],
            "x": boss_x,
            "y": boss_y,
            "hp": bdef["hp"],
            "max_hp": bdef["hp"],
            "is_boss": True,
            "floor": floor,
            "attack_cooldown": 0,
        }
        self._mobs.append(boss_mob)
        self._bosses_alive[floor] = True

        # Scatter dungeon-specific resources on floor cells.
        num_res = int(self.rng.integers(3, 6))
        for _ in range(num_res):
            res = str(self.rng.choice([TILE_STONE, TILE_COAL, TILE_IRON]))
            for _att in range(20):
                rx = int(self.rng.integers(1, size - 1))
                ry = int(self.rng.integers(1, size - 1))
                if grid[ry][rx] == TILE_DUNGEON_FLOOR:
                    grid[ry][rx] = res
                    break

        # T20β: Floor 4 (Vaults) — place 1 TILE_ENCHANT_FIRE tile.
        # We scan the grid in row-major order and place it on the first
        # TILE_DUNGEON_FLOOR cell that is not a stair or reserved position.
        # This gives a deterministic location for a given seed.
        if floor == 4:
            _reserved = {stairs_up_pos, stairs_down_pos}
            _placed_enchant = False
            for _ey in range(size):
                if _placed_enchant:
                    break
                for _ex in range(size):
                    if (
                        grid[_ey][_ex] == TILE_DUNGEON_FLOOR
                        and (_ex, _ey) not in _reserved
                    ):
                        grid[_ey][_ex] = TILE_ENCHANT_FIRE
                        _placed_enchant = True
                        break

        # Spawn dungeon mobs (avoid tiles already occupied).
        mob_types = ["skeleton", "kobold", "bat"]
        num_mobs = floor + 2
        for _ in range(num_mobs):
            mtype = str(self.rng.choice(mob_types))
            for _att in range(30):
                mx = int(self.rng.integers(1, size - 1))
                my = int(self.rng.integers(1, size - 1))
                if (
                    grid[my][mx] == TILE_DUNGEON_FLOOR
                    and not self._mob_at(mx, my, floor)
                ):
                    stats = _MOB_STATS[mtype]
                    mob: Mob = {
                        "type": mtype,
                        "x": mx,
                        "y": my,
                        "hp": stats["hp"],
                        "max_hp": stats["hp"],
                        "is_boss": False,
                        "floor": floor,
                        "attack_cooldown": 0,
                    }
                    self._mobs.append(mob)
                    break

        self._floors[floor] = grid
        self._torches[floor] = set()

    # ---------------------------------------------------------------
    # Cow spawning
    # ---------------------------------------------------------------

    def _spawn_initial_cows(self) -> None:
        num_cows = int(self.rng.integers(3, 6))
        size = _SURFACE_SIZE
        for _ in range(num_cows):
            for _att in range(50):
                x = int(self.rng.integers(5, size - 5))
                y = int(self.rng.integers(5, size - 5))
                if (
                    self._floors[0][y][x] == TILE_GRASS
                    and not self._mob_at(x, y, 0)
                ):
                    mob: Mob = {
                        "type": "cow",
                        "x": x, "y": y,
                        "hp": _MOB_STATS["cow"]["hp"],
                        "max_hp": _MOB_STATS["cow"]["hp"],
                        "is_boss": False,
                        "floor": 0,
                        "attack_cooldown": 0,
                    }
                    self._mobs.append(mob)
                    break

    # ---------------------------------------------------------------
    # Adjacency helpers
    # ---------------------------------------------------------------

    def _near_table(self) -> bool:
        grid = self._current_grid()
        fsize = self._floor_size()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = self._agent_x + dx, self._agent_y + dy
            if (
                0 <= nx < fsize
                and 0 <= ny < fsize
                and grid[ny][nx] == TILE_TABLE
            ):
                return True
        return False

    def _near_furnace(self) -> bool:
        grid = self._current_grid()
        fsize = self._floor_size()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = self._agent_x + dx, self._agent_y + dy
            if (
                0 <= nx < fsize
                and 0 <= ny < fsize
                and grid[ny][nx] == TILE_FURNACE
            ):
                return True
        return False

    # ---------------------------------------------------------------
    # Achievement system
    # ---------------------------------------------------------------

    def _try_unlock(self, name: str) -> float:
        if (
            name in self._ALL_ACHIEVEMENTS
            and name not in self._achievements_unlocked
        ):
            self._achievements_unlocked.add(name)
            pretty = name.replace("_", " ").title()
            self._message = f"ACHIEVEMENT: {pretty}!"
            return 1.0
        return 0.0

    # ---------------------------------------------------------------
    # Mob helpers
    # ---------------------------------------------------------------

    def _mob_at(
        self, x: int, y: int, floor: int | None = None,
    ) -> Mob | None:
        fl = floor if floor is not None else self._current_floor
        for mob in self._mobs:
            if (
                mob["x"] == x
                and mob["y"] == y
                and mob["floor"] == fl
            ):
                return mob
        return None

    def _best_weapon_bonus(self) -> int:
        best = 0
        for weapon, bonus in _WEAPON_BONUS.items():
            if self._inventory.get(weapon, 0) > 0 and bonus > best:
                best = bonus
        if self._weapon_enchanted:
            best += 2
        return best

    def _best_armor_defense(self) -> int:
        best = 0
        for armor, defense in _ARMOR_DEFENSE.items():
            if self._inventory.get(armor, 0) > 0 and defense > best:
                best = defense
        if self._armor_enchanted:
            best += 1
        return best

    def _take_damage(self, raw: int) -> None:
        defense = self._best_armor_defense()
        # Phase α: 3.5× damage multiplier while sleeping (upstream
        # game_logic.py:1100-1291). Mob projectiles also call this
        # path, so they inherit the multiplier consistently.
        if self._is_sleeping:
            raw = int(round(raw * 3.5))
        actual = max(1, raw - defense)
        self._hp = max(0, self._hp - actual)
        # Phase β: damage cancels REST and SLEEP states.
        self._is_resting = False
        self._is_sleeping = False

    def _attack_mob(self, mob: Mob) -> float:
        damage = 1 + self._best_weapon_bonus()
        mob["hp"] -= damage
        reward = 0.0
        if mob["hp"] <= 0:
            self._mobs.remove(mob)
            self._total_kills += 1
            reward += self._check_kill_milestones()
            mtype = mob["type"]
            if mob["is_boss"]:
                fl = mob["floor"]
                self._bosses_alive[fl] = False
                bdef = _BOSS_DEFS.get(fl)
                if bdef:
                    self._message = (
                        f"Defeated {bdef['name']}!"
                    )
                    reward += self._try_unlock(bdef["ach"])
                    # Drop boss loot
                    loot_key = f"boss_loot_{fl}"
                    self._inventory[loot_key] = (
                        self._inventory.get(loot_key, 0) + 1
                    )
                    ach = f"collect_boss_loot_{fl}"
                    reward += self._try_unlock(ach)
                    # No-armor boss kill
                    if self._best_armor_defense() == 0:
                        reward += self._try_unlock(
                            "defeat_boss_no_armor"
                        )
            elif mtype == "zombie":
                self._message = "Defeated a zombie!"
                reward += self._try_unlock("defeat_zombie")
            elif mtype == "skeleton":
                self._message = "Defeated a skeleton!"
                reward += self._try_unlock("defeat_skeleton")
            elif mtype == "cow":
                self._food = min(_MAX_FOOD, self._food + 5)
                self._message = (
                    "Defeated a cow! Ate beef. (+5 food)"
                )
                reward += self._try_unlock("eat_cow")
            elif mtype == "kobold":
                self._message = "Defeated a kobold!"
                reward += self._try_unlock("defeat_kobold")
            elif mtype == "bat":
                self._message = "Defeated a bat!"
                reward += self._try_unlock("defeat_bat")
        else:
            self._message = (
                f"Hit {mob['type']}! "
                f"({mob['hp']}/{mob['max_hp']} HP)"
            )
        return reward

    def _check_kill_milestones(self) -> float:
        reward = 0.0
        if self._total_kills >= 5:
            reward += self._try_unlock("kill_5_mobs")
        if self._total_kills >= 10:
            reward += self._try_unlock("kill_10_mobs")
        if self._total_kills >= 25:
            reward += self._try_unlock("kill_25_mobs")
        if self._total_kills >= 50:
            reward += self._try_unlock("kill_50_mobs")
        return reward

    def _check_exploration_milestones(self) -> float:
        r = 0.0
        # All floors visited
        all_floor_achs = {
            "enter_dungeon", "reach_floor_2",
            "reach_floor_3", "reach_floor_4",
            "reach_floor_5",
        }
        if all_floor_achs.issubset(
            self._achievements_unlocked
        ):
            r += self._try_unlock("explore_all_floors")
        # All boss loot collected
        all_loot = all(
            f"collect_boss_loot_{i}"
            in self._achievements_unlocked
            for i in range(1, 6)
        )
        if all_loot:
            r += self._try_unlock("collect_all_boss_loot")
        # Spells learned — all spells known
        if all(self._learned_spells.values()):
            r += self._try_unlock("learn_all_spells")
        # Clear dungeon floor (no mobs on current floor)
        if self._current_floor > 0:
            floor_mobs = [
                m for m in self._mobs
                if m["floor"] == self._current_floor
            ]
            if len(floor_mobs) == 0:
                r += self._try_unlock(
                    "clear_dungeon_floor"
                )
        return r

    def _is_in_boss_fight(self) -> bool:
        """True iff a boss mob is alive on the agent's current floor."""
        return any(
            m["is_boss"] and m["floor"] == self._current_floor and m["hp"] > 0
            for m in self._mobs
        )

    def _step_player_projectiles(self) -> None:
        """Advance live player projectiles one tile and resolve hits.

        - Map bounds: drop projectiles that leave the floor.
        - Solid-block collision: projectiles dying on solid tiles do not damage.
        - Mob hit: damage the first mob the projectile lands on; projectile dies.
        Mirrors upstream _move_player_projectile:1719-1823 (post-advance check
        only; pre-advance dual-position check is deferred to phase γ).

        blocked_fn and hit_fn both receive the ProjectileEntity directly.
        This avoids the stale next()-lookup bug where two projectiles advancing
        to the same tile would both resolve to the first one in the list.
        pending_kills is a local variable (not an instance attribute) so a raise
        inside _attack_mob_kill cannot leave a dangling attribute.
        """
        if not self._player_projectiles:
            return
        from glyphbench.envs.craftax.mechanics.projectiles import (
            step_player_projectiles,
        )

        size = self._floor_size()
        pending_kills: list = []  # local, no instance attr

        def _blocked(p) -> bool:
            return self._tile_at(p.x, p.y) in _SOLID_TILES

        def _hit(p) -> bool:
            for mob in self._mobs:
                if (
                    mob["floor"] == self._current_floor
                    and mob["x"] == p.x
                    and mob["y"] == p.y
                    and mob["hp"] > 0
                ):
                    mob["hp"] -= p.damage
                    if mob["hp"] <= 0:
                        pending_kills.append(mob)
                    return True
            return False

        self._player_projectiles = step_player_projectiles(
            self._player_projectiles,
            map_w=size, map_h=size,
            blocked_fn=_blocked, hit_fn=_hit,
        )

        # Resolve any pending kills (fire achievement, accumulate reward).
        # _attack_mob_kill removes the mob from self._mobs itself.
        for dead_mob in pending_kills:
            self._pending_step_reward += self._attack_mob_kill(dead_mob)

    def _step_mob_projectiles(self) -> None:
        """Advance live mob projectiles; damage player on hit; destroy
        furnace/crafting_table on impact; cancel sleep when hit.

        Symmetric to _step_player_projectiles (T12). The order of the
        per-step driver matters: T12 (player projectiles) runs BEFORE
        _mob_ai; T25 (mob projectiles) runs AFTER _mob_ai so projectiles
        spawned by ranged mobs in this tick travel immediately.
        """
        if not self._mob_projectiles:
            return
        from glyphbench.envs.craftax.mechanics.projectiles import (
            step_mob_projectiles,
        )
        size = self._floor_size()

        def _blocked(p) -> bool:
            return self._tile_at(p.x, p.y) in _SOLID_TILES

        def _block_destroy(p) -> bool:
            tile = self._tile_at(p.x, p.y)
            if tile in (TILE_TABLE, TILE_FURNACE):
                # Replace with floor tile (per upstream behaviour).
                empty = (
                    TILE_GRASS if self._current_floor == 0
                    else TILE_DUNGEON_FLOOR
                )
                self._current_grid()[p.y][p.x] = empty
                return True
            return False

        def _hit_player(p) -> bool:
            if p.x == self._agent_x and p.y == self._agent_y:
                self._take_damage(p.damage)
                self._is_sleeping = False  # upstream: hit cancels sleep
                return True
            return False

        self._mob_projectiles = step_mob_projectiles(
            self._mob_projectiles,
            map_w=size, map_h=size,
            blocked_fn=_blocked,
            block_destruction_fn=_block_destroy,
            hit_player_fn=_hit_player,
        )

    def _mob_ai(self) -> None:
        """Move mobs on current floor and handle attacks.

        Phase α/β: melee mobs (zombie) use the cooldown-aware
        step_melee_mob from mechanics/mobs.py. Ranged mobs (skeleton, kobold)
        use step_ranged_mob with kiting AI + cooldown=4 + projectile spawn
        (T24). Passive (cow, bat) mobs still use the inline logic
        until a future passive-AI task.

        Turn order: attack-then-move per upstream. step_melee_mob and
        step_ranged_mob encapsulate that; the inline branches mirror it for
        non-melee/non-ranged.
        """
        from glyphbench.envs.craftax.mechanics.mobs import step_melee_mob

        fsize = self._floor_size()
        walkable = self._walkable_set()

        def _is_blocked(x: int, y: int) -> bool:
            return not (
                0 <= x < fsize
                and 0 <= y < fsize
                and (x, y) != (self._agent_x, self._agent_y)
                and self._current_grid()[y][x] in walkable
            )

        def _damage_for(mob: dict) -> int:
            mtype = mob["type"]
            if mob.get("is_boss"):
                bdef = _BOSS_DEFS.get(mob["floor"], {})
                return bdef.get("damage", 3)
            return _MOB_STATS.get(mtype, {"damage": 1})["damage"]

        for mob in list(self._mobs):
            if mob["floor"] != self._current_floor:
                continue
            mx, my = mob["x"], mob["y"]
            mtype = mob["type"]

            # Phase-α/β melee mobs: only zombie (upstream-faithful floor-0 melee).
            if mtype == "zombie":
                step_melee_mob(
                    mob,
                    player_x=self._agent_x,
                    player_y=self._agent_y,
                    is_blocked_for_mob=_is_blocked,
                    apply_damage_to_player=self._take_damage,
                    rng=random.Random(int(self.rng.integers(0, 2**31))),
                    is_fighting_boss=self._is_in_boss_fight(),
                    damage_for_mob=_damage_for,
                )
                continue

            # Phase-β ranged mobs: skeleton (fires arrows) and kobold (throws daggers).
            # T_FOLLOWUP_A / T04β: skeleton_archer → skeleton, spider → kobold.
            elif mtype in ("skeleton", "kobold"):
                from glyphbench.envs.craftax.mechanics.projectiles import (
                    ProjectileEntity,
                    ProjectileType,
                )
                from glyphbench.envs.craftax.mechanics.mobs import step_ranged_mob

                def _spawn_proj(kind, x, y, dx, dy, dmg):
                    self._mob_projectiles.append(
                        ProjectileEntity(kind=kind, x=x, y=y, dx=dx, dy=dy, damage=dmg)
                    )

                # skeleton fires arrows (ARROW2), kobold throws daggers (DAGGER)
                def _proj_kind(m: dict) -> ProjectileType:
                    return (
                        ProjectileType.ARROW2
                        if m["type"] == "skeleton"
                        else ProjectileType.DAGGER
                    )

                step_ranged_mob(
                    mob,
                    player_x=self._agent_x,
                    player_y=self._agent_y,
                    is_blocked_for_mob=_is_blocked,
                    spawn_mob_projectile=_spawn_proj,
                    rng=random.Random(int(self.rng.integers(0, 2**31))),
                    max_mob_projectiles_room=3 - sum(
                        1 for _ in self._mob_projectiles
                    ),
                    projectile_kind_for_mob=_proj_kind,
                    damage_for_mob=_damage_for,
                )
                continue

            # === existing inline logic for everything else === #

            # 1. ATTACK (hostile mobs only). Adjacency is measured at
            # mob's pre-move position so the player escaping melee on
            # the agent's turn breaks the damage chain this turn.
            if mtype != "cow":
                start_adj = (
                    abs(mx - self._agent_x) + abs(my - self._agent_y)
                )
                if mob["is_boss"]:
                    bdef = _BOSS_DEFS.get(mob["floor"], {})
                    dmg = bdef.get("damage", 3)
                    is_ranged = bdef.get("ranged", False)
                else:
                    dmg = _MOB_STATS.get(
                        mtype, {"damage": 1}
                    )["damage"]
                    is_ranged = False  # inline branch only handles cow/bat; ranged mobs have step_ranged_mob above

                if start_adj <= 1:
                    self._take_damage(dmg)
                    if not self._message:
                        self._message = (
                            f"A {mtype} hits you for {dmg}!"
                        )
                elif is_ranged and start_adj <= 4:
                    self._take_damage(max(1, dmg // 2))
                    if not self._message:
                        self._message = (
                            f"A {mtype} shoots you!"
                        )

            # 2. MOVE.
            ddx, ddy = 0, 0
            if mtype == "cow":
                d = int(self.rng.integers(0, 5))
                ddx, ddy = [
                    (0, 0), (1, 0), (-1, 0),
                    (0, 1), (0, -1),
                ][d]
            elif mtype == "bat":
                d = int(self.rng.integers(0, 4))
                ddx, ddy = [
                    (1, 0), (-1, 0), (0, 1), (0, -1),
                ][d]
            else:
                dist_x = self._agent_x - mx
                dist_y = self._agent_y - my
                if abs(dist_x) + abs(dist_y) <= 8:
                    if abs(dist_x) >= abs(dist_y):
                        ddx = (
                            1 if dist_x > 0
                            else (-1 if dist_x < 0 else 0)
                        )
                    else:
                        ddy = (
                            1 if dist_y > 0
                            else (-1 if dist_y < 0 else 0)
                        )

            nx, ny = mx + ddx, my + ddy
            if (
                0 <= nx < fsize
                and 0 <= ny < fsize
                and (nx, ny) != (self._agent_x, self._agent_y)
                and not self._mob_at(nx, ny)
                and self._current_grid()[ny][nx] in walkable
            ):
                mob["x"] = nx
                mob["y"] = ny

    def _spawn_night_mobs(self) -> None:
        if self._current_floor != 0:
            return
        num = int(self.rng.integers(2, 5))
        lm = self._lightmap.get(0)
        for _ in range(num):
            mt = "zombie"  # upstream floor-0 melee mob only (T_FOLLOWUP_A)
            for _att in range(20):
                dx = int(self.rng.integers(-6, 7))
                dy = int(self.rng.integers(-6, 7))
                x = self._agent_x + dx
                y = self._agent_y + dy
                dist = abs(dx) + abs(dy)
                if (
                    3 <= dist <= 6
                    and 0 <= x < _SURFACE_SIZE
                    and 0 <= y < _SURFACE_SIZE
                    and self._floors[0][y][x]
                    in SURFACE_WALKABLE
                    and not self._mob_at(x, y, 0)
                    and (x, y)
                    != (self._agent_x, self._agent_y)
                ):
                    # Phase β T17β: scale spawn chance by (1 - light_level)².
                    # Darker tiles attract more mobs quadratically (upstream mechanic).
                    light = float(lm[y, x]) if lm is not None else 0.0
                    effective_chance = _NIGHT_SPAWN_BASE_CHANCE * (1.0 - light) ** 2
                    if self.rng.random() < effective_chance:
                        mob: Mob = {
                            "type": mt,
                            "x": x, "y": y,
                            "hp": _MOB_STATS[mt]["hp"],
                            "max_hp": _MOB_STATS[mt]["hp"],
                            "is_boss": False,
                            "floor": 0,
                            "attack_cooldown": 0,
                        }
                        self._mobs.append(mob)
                    break

    def _despawn_night_mobs(self) -> None:
        self._mobs = [
            m for m in self._mobs
            if m["floor"] != 0
            or m["type"] != "zombie"  # only zombie spawns at night on floor 0 (T_FOLLOWUP_A)
        ]

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # Lightmap (phase β T15β + T16β)
    # ---------------------------------------------------------------

    def _biome_baseline(self, floor: int) -> float:
        """Return the per-floor ambient light baseline in [0, 1].

        Floor 0 (overworld): 1.0 during day, 0.3 at night (phase β binary).
        Floors 1-5 (dungeons/mines): 0.0 (requires torches).
        """
        if floor == 0:
            return 1.0 if self._day_night == "day" else 0.3
        return 0.0

    def _recompute_lightmap(self, floor: int) -> None:
        """Recompute the lightmap for *floor* from torches + biome baseline."""
        from glyphbench.envs.craftax.mechanics.lighting import compute_lightmap
        grid = self._floors.get(floor)
        if grid is None:
            return
        grid_h = len(grid)
        grid_w = len(grid[0]) if grid_h > 0 else 0
        torches = self._torches.get(floor, set())
        baseline = self._biome_baseline(floor)
        self._lightmap[floor] = compute_lightmap(grid_h, grid_w, torches, baseline)

    # ---------------------------------------------------------------
    # Day/night cycle
    # ---------------------------------------------------------------

    def _advance_day_counter(self, steps: int = 1) -> None:
        for _ in range(steps):
            old = self._day_night
            self._day_counter += 1
            pos = self._day_counter % _CYCLE_LENGTH
            new = "day" if pos < _DAY_LENGTH else "night"
            self._day_night = new

            if old == "day" and new == "night":
                self._night_count += 1
                self._message = (
                    "The sun sets. Monsters emerge."
                )
                self._spawn_night_mobs()
                # Night survival milestones
                if self._night_count >= 5:
                    self._try_unlock("survive_5_nights")
                if self._night_count >= 10:
                    self._try_unlock("survive_10_nights")
                if self._night_count >= 20:
                    self._try_unlock("survive_20_nights")
                # Recompute floor-0 lightmap on day->night flip.
                self._recompute_lightmap(0)
            elif old == "night" and new == "day":
                self._message = "The sun rises."
                self._despawn_night_mobs()
                # Recompute floor-0 lightmap on night->day flip.
                self._recompute_lightmap(0)

    # ---------------------------------------------------------------
    # Survival mechanics
    # ---------------------------------------------------------------

    def _apply_survival_drain(self) -> None:
        if self._food == 0:
            self._hp -= 1
        if self._water == 0:
            self._hp -= 1
        self._hp = max(0, self._hp)

        step = self._day_counter
        if step > 0 and step % _FOOD_DRAIN_INTERVAL == 0:
            self._food = max(0, self._food - 1)
        if step > 0 and step % _WATER_DRAIN_INTERVAL == 0:
            self._water = max(0, self._water - 1)
        if step > 0 and step % _ENERGY_DRAIN_INTERVAL == 0:
            self._energy = max(0, self._energy - 1)
        # Mana regen
        if step > 0 and step % _MANA_REGEN_INTERVAL == 0:
            self._mana = min(_MAX_MANA, self._mana + 1)

        # Tick effects
        if self._speed_turns > 0:
            self._speed_turns -= 1

    def _tick_plants(self) -> None:
        to_ripen: list[tuple[int, int]] = []
        for pos, remaining in self._plants.items():
            remaining -= 1
            self._plants[pos] = remaining
            if remaining <= 0:
                to_ripen.append(pos)
        for pos in to_ripen:
            del self._plants[pos]
            x, y = pos
            grid = self._floors[0]
            if (
                0 <= x < _SURFACE_SIZE
                and 0 <= y < _SURFACE_SIZE
                and grid[y][x] == TILE_SAPLING
            ):
                grid[y][x] = TILE_RIPE_PLANT

    def _has_shelter(self) -> bool:
        grid = self._current_grid()
        fsize = self._floor_size()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx = self._agent_x + dx
            ny = self._agent_y + dy
            if not (
                0 <= nx < fsize
                and 0 <= ny < fsize
                and grid[ny][nx] == TILE_PLACED_STONE
            ):
                return False
        return True

    # ---------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._floors = {}
        self._mobs = []
        self._player_projectiles = []
        self._mob_projectiles = []
        self._is_sleeping = False
        self._is_resting = False
        self._pending_step_reward: float = 0.0
        self._torches = {0: set()}
        self._lightmap = {}
        self._stairs_down_pos = {}
        self._stairs_up_pos = {}
        self._bosses_alive = {}
        self._chests_opened = {}
        self._first_chest_opened = {}

        self._generate_surface()
        for fl in range(1, _NUM_DUNGEON_FLOORS + 1):
            self._generate_dungeon_floor(fl)

        self._current_floor = 0
        self._agent_x = _SURFACE_SIZE // 2
        self._agent_y = _SURFACE_SIZE // 2
        self._facing = (1, 0)
        self._inventory = {
            "bow": 0,
            "arrows": 0,
            "torch": 0,
            "sapphire": 0,
            "ruby": 0,
            "book": 0,
            "potions": {"red": 0, "green": 0, "blue": 0, "pink": 0, "cyan": 0, "yellow": 0},
        }
        # Phase β T08β: hidden per-episode color->effect mapping.
        from glyphbench.envs.craftax.mechanics.potions import make_potion_mapping
        self._potion_mapping = make_potion_mapping(seed)
        self._achievements_unlocked = set()
        self._achievements_phase_beta = {n: False for n in UPSTREAM_ACHIEVEMENT_NAMES}
        self._message = ""
        self._hp = self._max_hp
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        self._learned_spells = {"fireball": False, "iceball": False}
        self._day_counter = 0
        self._day_night = "day"
        self._night_count = 0
        self._plants = {}
        self._potions = []  # legacy list — unused in phase β
        self._speed_turns = 0
        self._weapon_enchanted = False
        self._armor_enchanted = False
        self._total_kills = 0
        self._total_crafts = 0
        self._total_blocks_placed = 0
        self._total_plants_eaten = 0
        self._total_water_drunk = 0

        self._spawn_initial_cows()
        # Compute initial lightmaps for all generated floors.
        for fl in self._floors:
            self._recompute_lightmap(fl)
        return self._render_current_observation()

    # ---------------------------------------------------------------
    # Step
    # ---------------------------------------------------------------

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        self._message = ""
        reward = 0.0

        # Phase β: REST state machine — regen HP before action handler runs.
        if self._is_resting:
            self._hp = min(self._max_hp, self._hp + 1)
            # Check exit conditions.
            if (
                self._hp >= self._max_hp
                or self._food <= 0
                or self._water <= 0
            ):
                self._is_resting = False

        # Phase β: SLEEP state machine — continuous sleep with +2 HP/tick and
        # +2 energy/tick.  Runs BEFORE the action handler so that each tick
        # while sleeping applies regen regardless of what action is dispatched
        # (action handlers are effectively NOOPs while sleeping because the
        # agent should not issue another SLEEP while sleeping, and any movement
        # etc. issued by an external driver is irrelevant — the sleep loop is
        # the intended game-play path).
        if self._is_sleeping:
            self._hp = min(self._max_hp, self._hp + 2)
            self._energy = min(_MAX_ENERGY, self._energy + 2)
            if self._energy >= _MAX_ENERGY:
                self._is_sleeping = False
                reward += self._try_unlock("wake_up")

        # Phase β T07β: lava damage tick — applied BEFORE the action handler so
        # damage state (HP, REST/SLEEP cancel via _take_damage) is already
        # updated before any movement this tick.
        if (
            self._is_in_bounds(self._agent_x, self._agent_y)
            and self._tile_at(self._agent_x, self._agent_y) == TILE_LAVA
        ):
            self._take_damage(2)

        handler = self._ACTION_DISPATCH.get(name)
        if handler is not None:
            reward += handler(self)
        # (NOOP falls through with 0 reward)

        # Post-action ticks
        self._advance_day_counter()
        self._apply_survival_drain()
        self._tick_plants()
        # Phase α — advance player projectiles after the action handler runs.
        self._step_player_projectiles()
        reward += self._pending_step_reward
        self._pending_step_reward = 0.0
        self._mob_ai()
        # Phase α — advance mob projectiles after _mob_ai so projectiles
        # spawned by ranged mobs in this tick travel immediately (T25).
        self._step_mob_projectiles()

        # Phase α: despawn mobs that drifted beyond MOB_DESPAWN_DISTANCE.
        from glyphbench.envs.craftax.mechanics.mobs import should_despawn
        is_boss_fight = self._is_in_boss_fight()
        self._mobs = [
            m for m in self._mobs
            if not (
                m["floor"] == self._current_floor
                and should_despawn(
                    m,
                    player_x=self._agent_x,
                    player_y=self._agent_y,
                    is_fighting_boss=is_boss_fight,
                )
            )
        ]

        # Check stat milestones
        if self._hp == self._max_hp:
            reward += self._try_unlock("full_health")
        if self._mana == _MAX_MANA:
            reward += self._try_unlock("full_mana")

        # Exploration milestones
        reward += self._check_exploration_milestones()

        terminated = self._hp <= 0
        if terminated:
            self._message = "You died."

        info: dict[str, Any] = {
            "agent_pos": (self._agent_x, self._agent_y),
            "floor": self._current_floor,
            "inventory": dict(self._inventory),
            "achievements": list(self._achievements_unlocked),
            "achievements_this_step": (
                [self._message.split(": ")[1].rstrip("!")]
                if self._message.startswith("ACHIEVEMENT")
                else []
            ),
            "day_night": self._day_night,
            "food": self._food,
            "water": self._water,
            "energy": self._energy,
            "mana": self._mana,
        }

        obs = self._render_current_observation()
        return obs, reward, terminated, False, info

    # ---------------------------------------------------------------
    # Action handlers
    # ---------------------------------------------------------------

    def _handle_move(self, name: str) -> float:
        direction_map: dict[str, tuple[int, int]] = {
            "MOVE_LEFT": (-1, 0),
            "MOVE_RIGHT": (1, 0),
            "MOVE_UP": (0, -1),
            "MOVE_DOWN": (0, 1),
        }
        dx, dy = direction_map[name]
        self._facing = (dx, dy)

        if self._energy == 0 and self.rng.random() < 0.5:
            self._message = "Too exhausted to move!"
            return 0.0

        steps = 2 if self._speed_turns > 0 else 1
        for _ in range(steps):
            nx = self._agent_x + dx
            ny = self._agent_y + dy
            fsize = self._floor_size()
            grid = self._current_grid()
            walkable = self._walkable_set()
            if (
                0 <= nx < fsize
                and 0 <= ny < fsize
                and grid[ny][nx] in walkable
                and not self._mob_at(nx, ny)
            ):
                self._agent_x = nx
                self._agent_y = ny
        return 0.0

    def _handle_move_left(self) -> float:
        return self._handle_move("MOVE_LEFT")

    def _handle_move_right(self) -> float:
        return self._handle_move("MOVE_RIGHT")

    def _handle_move_up(self) -> float:
        return self._handle_move("MOVE_UP")

    def _handle_move_down(self) -> float:
        return self._handle_move("MOVE_DOWN")

    def _handle_do(self) -> float:
        fsize = self._floor_size()
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        if not (0 <= fx < fsize and 0 <= fy < fsize):
            return 0.0

        reward = 0.0
        mob = self._mob_at(fx, fy)
        if mob is not None:
            reward += self._attack_mob(mob)
            return reward

        grid = self._current_grid()
        tile = grid[fy][fx]

        # ---- Chest interaction (T12β / T13β / T14β) ----
        if tile == TILE_CHEST:
            floor = self._current_floor
            opened_on_floor = self._chests_opened.get(floor, set())
            if (fx, fy) in opened_on_floor:
                self._message = "This chest is already open."
                return 0.0
            reward += self._open_chest(fx, fy, floor)
            return reward

        # ---- Fountain interaction (T18β) ----
        # Drinking from a fountain refills water to max. The tile is NOT
        # consumed — fountains are reusable.
        if tile == TILE_FOUNTAIN:
            if self._water >= _MAX_WATER:
                self._message = "The fountain is full — you are not thirsty."
                return 0.0
            self._water = _MAX_WATER
            self._total_water_drunk += 1
            self._message = "You drink deeply from the fountain. Water restored!"
            reward += self._try_unlock("collect_drink")
            if self._total_water_drunk >= 10:
                reward += self._try_unlock("drink_10_water")
            return reward

        if tile in INTERACTABLE_TILES:
            resource = INTERACTABLE_TILES[tile]
            if tile in PICKAXE_REQUIRED:
                min_tier = PICKAXE_REQUIRED[tile]
                has_pick = any(
                    self._inventory.get(p, 0) > 0
                    for p in _PICKAXE_TIERS[min_tier:]
                )
                if not has_pick:
                    needed = _PICKAXE_TIERS[min_tier]
                    self._message = (
                        f"Need a {needed.replace('_', ' ')}."
                    )
                    return 0.0
            self._inventory[resource] = (
                self._inventory.get(resource, 0) + 1
            )
            # Replace with floor tile
            if self._current_floor == 0:
                grid[fy][fx] = TILE_GRASS
            else:
                grid[fy][fx] = TILE_DUNGEON_FLOOR
            self._message = f"Collected {resource}."
            reward += self._try_unlock(
                f"collect_{resource}"
            )
            if (
                tile == TILE_TREE
                and self.rng.random() < _SAPLING_DROP_CHANCE
            ):
                self._inventory["sapling"] = (
                    self._inventory.get("sapling", 0) + 1
                )
                self._message += " Found a sapling!"
                reward += self._try_unlock(
                    "collect_sapling"
                )
            # Resource milestones
            reward += self._check_resource_milestones()
        return reward

    # Pickaxe and sword tiers ordered by level (1-indexed as in upstream).
    _PICKAXE_BY_LEVEL: tuple[str, ...] = (
        "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe",
    )
    _SWORD_BY_LEVEL: tuple[str, ...] = (
        "wood_sword", "stone_sword", "iron_sword", "diamond_sword",
    )

    def _open_chest(self, fx: int, fy: int, floor: int) -> float:
        """Roll loot from a chest and apply first-chest gating (T13β + T14β).

        Parameters
        ----------
        fx, fy : int
            Position of the chest tile on the current floor grid.
        floor : int
            Current dungeon floor number.
        """
        reward = 0.0
        loot_parts: list[str] = []

        # ---- Generic loot table (T13β) ----
        # Wood (60 %)
        if self.rng.random() < 0.6:
            amount = int(self.rng.integers(1, 6))  # 1-5 inclusive
            self._inventory["wood"] = self._inventory.get("wood", 0) + amount
            loot_parts.append(f"{amount} wood")

        # Torches (60 %)
        if self.rng.random() < 0.6:
            amount = int(self.rng.integers(4, 8))  # 4-7 inclusive
            self._inventory["torch"] = self._inventory.get("torch", 0) + amount
            loot_parts.append(f"{amount} torches")

        # Ore (60 %, mutually exclusive)
        if self.rng.random() < 0.6:
            ore_roll = self.rng.random()
            # Probabilities: coal 30 %, iron 30 %, diamond 15 %,
            # sapphire 12.5 %, ruby 12.5 %
            if ore_roll < 0.30:
                amount = int(self.rng.integers(1, 4))  # 1-3
                self._inventory["coal"] = self._inventory.get("coal", 0) + amount
                loot_parts.append(f"{amount} coal")
            elif ore_roll < 0.60:
                amount = int(self.rng.integers(1, 3))  # 1-2
                self._inventory["iron"] = self._inventory.get("iron", 0) + amount
                loot_parts.append(f"{amount} iron")
            elif ore_roll < 0.75:
                self._inventory["diamond"] = self._inventory.get("diamond", 0) + 1
                loot_parts.append("1 diamond")
            elif ore_roll < 0.875:
                self._inventory["sapphire"] = self._inventory.get("sapphire", 0) + 1
                loot_parts.append("1 sapphire")
            else:
                self._inventory["ruby"] = self._inventory.get("ruby", 0) + 1
                loot_parts.append("1 ruby")

        # Potions (50 %)
        if self.rng.random() < 0.5:
            _potion_colors = ("red", "green", "blue", "pink", "cyan", "yellow")
            color_idx = int(self.rng.integers(0, 6))
            amount = int(self.rng.integers(1, 3))  # 1-2
            color = _potion_colors[color_idx]
            potions = self._inventory.get("potions", {})
            potions[color] = potions.get(color, 0) + amount
            self._inventory["potions"] = potions
            loot_parts.append(f"{amount} {color} potion(s)")

        # Arrows (25 %)
        if self.rng.random() < 0.25:
            amount = int(self.rng.integers(1, 5))  # 1-4
            self._inventory["arrows"] = self._inventory.get("arrows", 0) + amount
            loot_parts.append(f"{amount} arrows")

        # Pickaxe upgrade (20 %)
        if self.rng.random() < 0.2:
            level_roll = self.rng.random()
            if level_roll < 0.40:
                level = 1
            elif level_roll < 0.70:
                level = 2
            elif level_roll < 0.90:
                level = 3
            else:
                level = 4
            new_key = self._PICKAXE_BY_LEVEL[level - 1]
            # Keep max: only grant if this is better than what we have.
            current_best = 0
            for li, pk in enumerate(self._PICKAXE_BY_LEVEL):
                if self._inventory.get(pk, 0) > 0:
                    current_best = li + 1
            if level > current_best:
                self._inventory[new_key] = self._inventory.get(new_key, 0) + 1
                loot_parts.append(f"{new_key.replace('_', ' ')}")

        # Sword upgrade (20 %)
        if self.rng.random() < 0.2:
            level_roll = self.rng.random()
            if level_roll < 0.40:
                level = 1
            elif level_roll < 0.70:
                level = 2
            elif level_roll < 0.90:
                level = 3
            else:
                level = 4
            new_key = self._SWORD_BY_LEVEL[level - 1]
            current_best = 0
            for li, sw in enumerate(self._SWORD_BY_LEVEL):
                if self._inventory.get(sw, 0) > 0:
                    current_best = li + 1
            if level > current_best:
                self._inventory[new_key] = self._inventory.get(new_key, 0) + 1
                loot_parts.append(f"{new_key.replace('_', ' ')}")

        # Mark chest as opened.
        self._chests_opened.setdefault(floor, set()).add((fx, fy))
        reward += self._try_unlock("open_chest")

        # ---- First-chest gating (T14β) ----
        if floor == 1 and not self._first_chest_opened.get(1):
            self._inventory["bow"] = max(self._inventory.get("bow", 0), 1)
            self._first_chest_opened[1] = True
            reward += self._try_unlock("find_bow")
            loot_parts.append("bow")

        if floor in (3, 4):
            already_got_book = (
                self._first_chest_opened.get(3)
                or self._first_chest_opened.get(4)
            )
            if not already_got_book:
                self._inventory["book"] = self._inventory.get("book", 0) + 1
                self._first_chest_opened[floor] = True
                reward += self._try_unlock("find_book")
                loot_parts.append("book")

        loot_str = ", ".join(loot_parts) if loot_parts else "nothing"
        self._message = f"Opened chest: {loot_str}."
        return reward

    def _check_resource_milestones(self) -> float:
        r = 0.0
        if self._inventory.get("wood", 0) >= 10:
            r += self._try_unlock("collect_10_wood")
        if self._inventory.get("wood", 0) >= 20:
            r += self._try_unlock("max_inventory_wood")
        if self._inventory.get("stone", 0) >= 5:
            r += self._try_unlock("collect_5_stone")
        if self._inventory.get("iron", 0) >= 3:
            r += self._try_unlock("collect_3_iron")
        return r

    def _handle_sleep(self) -> float:
        """Phase β: enters continuous sleep state. Regen and energy restore
        happen each tick in _step; this action just sets the flag."""
        self._is_sleeping = True
        self._message = "You fall asleep."
        return 0.0

    # -- Placement --

    def _handle_place_stone(self) -> float:
        fsize = self._floor_size()
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        empty = (
            TILE_GRASS if self._current_floor == 0
            else TILE_DUNGEON_FLOOR
        )
        if (
            0 <= fx < fsize
            and 0 <= fy < fsize
            and self._current_grid()[fy][fx] == empty
            and not self._mob_at(fx, fy)
            and self._inventory.get("stone", 0) >= 1
        ):
            self._inventory["stone"] -= 1
            self._current_grid()[fy][fx] = TILE_PLACED_STONE
            self._total_blocks_placed += 1
            self._message = "Placed stone."
            r = self._try_unlock("place_stone")
            if self._total_blocks_placed >= 10:
                r += self._try_unlock("place_10_blocks")
            return r
        return 0.0

    def _handle_place_table(self) -> float:
        fsize = self._floor_size()
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        empty = (
            TILE_GRASS if self._current_floor == 0
            else TILE_DUNGEON_FLOOR
        )
        if (
            0 <= fx < fsize
            and 0 <= fy < fsize
            and self._current_grid()[fy][fx] == empty
            and not self._mob_at(fx, fy)
            and self._inventory.get("wood", 0) >= 2
        ):
            self._inventory["wood"] -= 2
            self._current_grid()[fy][fx] = TILE_TABLE
            self._total_blocks_placed += 1
            self._message = "Placed crafting table."
            r = self._try_unlock("place_table")
            if self._total_blocks_placed >= 10:
                r += self._try_unlock("place_10_blocks")
            return r
        return 0.0

    def _handle_place_furnace(self) -> float:
        fsize = self._floor_size()
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        empty = (
            TILE_GRASS if self._current_floor == 0
            else TILE_DUNGEON_FLOOR
        )
        if (
            0 <= fx < fsize
            and 0 <= fy < fsize
            and self._current_grid()[fy][fx] == empty
            and not self._mob_at(fx, fy)
            and self._inventory.get("stone", 0) >= 4
        ):
            self._inventory["stone"] -= 4
            self._current_grid()[fy][fx] = TILE_FURNACE
            self._total_blocks_placed += 1
            self._message = "Placed furnace."
            r = self._try_unlock("place_furnace")
            if self._total_blocks_placed >= 10:
                r += self._try_unlock("place_10_blocks")
            return r
        return 0.0

    def _handle_place_plant(self) -> float:
        if self._current_floor != 0:
            return 0.0
        fsize = _SURFACE_SIZE
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        if (
            0 <= fx < fsize
            and 0 <= fy < fsize
            and self._floors[0][fy][fx] == TILE_GRASS
            and not self._mob_at(fx, fy)
            and self._inventory.get("sapling", 0) >= 1
        ):
            self._inventory["sapling"] -= 1
            self._floors[0][fy][fx] = TILE_SAPLING
            self._plants[(fx, fy)] = _PLANT_RIPEN_STEPS
            self._message = "Planted a sapling."
            return self._try_unlock("place_plant")
        return 0.0

    def _handle_place_torch(self) -> float:
        fsize = self._floor_size()
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        empty = (
            TILE_GRASS if self._current_floor == 0
            else TILE_DUNGEON_FLOOR
        )
        if (
            0 <= fx < fsize
            and 0 <= fy < fsize
            and self._current_grid()[fy][fx] == empty
            and self._inventory.get("torch", 0) >= 1
        ):
            self._inventory["torch"] -= 1
            self._current_grid()[fy][fx] = TILE_TORCH
            fl = self._current_floor
            if fl not in self._torches:
                self._torches[fl] = set()
            self._torches[fl].add((fx, fy))
            self._recompute_lightmap(fl)
            self._message = "Placed torch."
            return self._try_unlock("place_torch")
        return 0.0

    # -- Crafting: pickaxes --

    def _craft_increment(self) -> float:
        self._total_crafts += 1
        r = 0.0
        if self._total_crafts >= 10:
            r += self._try_unlock("craft_10_items")
        if self._total_crafts >= 25:
            r += self._try_unlock("craft_25_items")
        return r

    def _handle_make_wood_pickaxe(self) -> float:
        if (
            self._near_table()
            and self._inventory.get("wood", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["wood_pickaxe"] = (
                self._inventory.get("wood_pickaxe", 0) + 1
            )
            self._message = "Crafted wood pickaxe."
            r = self._try_unlock("make_wood_pickaxe")
            return r + self._craft_increment()
        return 0.0

    def _handle_make_stone_pickaxe(self) -> float:
        # Stone tier needs a table only (upstream Craftax recipe);
        # furnace is tier-3+ (iron and beyond).
        if (
            self._near_table()
            and self._inventory.get("wood", 0) >= 1
            and self._inventory.get("stone", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["stone"] -= 1
            self._inventory["stone_pickaxe"] = (
                self._inventory.get("stone_pickaxe", 0) + 1
            )
            self._message = "Crafted stone pickaxe."
            r = self._try_unlock("make_stone_pickaxe")
            return r + self._craft_increment()
        return 0.0

    def _handle_make_iron_pickaxe(self) -> float:
        if (
            self._near_table()
            and self._near_furnace()
            and self._inventory.get("wood", 0) >= 1
            and self._inventory.get("iron", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["iron"] -= 1
            self._inventory["iron_pickaxe"] = (
                self._inventory.get("iron_pickaxe", 0) + 1
            )
            self._message = "Crafted iron pickaxe."
            r = self._try_unlock("make_iron_pickaxe")
            return r + self._craft_increment()
        return 0.0

    def _handle_make_diamond_pickaxe(self) -> float:
        if (
            self._near_table()
            and self._near_furnace()
            and self._inventory.get("wood", 0) >= 1
            and self._inventory.get("diamond", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["diamond"] -= 1
            self._inventory["diamond_pickaxe"] = (
                self._inventory.get("diamond_pickaxe", 0) + 1
            )
            self._message = "Crafted diamond pickaxe."
            r = self._try_unlock("make_diamond_pickaxe")
            return r + self._craft_increment()
        return 0.0

    # -- Crafting: swords --

    def _handle_make_wood_sword(self) -> float:
        if (
            self._near_table()
            and self._inventory.get("wood", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["wood_sword"] = (
                self._inventory.get("wood_sword", 0) + 1
            )
            self._message = "Crafted wood sword."
            r = self._try_unlock("make_wood_sword")
            return r + self._craft_increment()
        return 0.0

    def _handle_make_stone_sword(self) -> float:
        # Stone tier needs a table only (upstream Craftax recipe).
        if (
            self._near_table()
            and self._inventory.get("wood", 0) >= 1
            and self._inventory.get("stone", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["stone"] -= 1
            self._inventory["stone_sword"] = (
                self._inventory.get("stone_sword", 0) + 1
            )
            self._message = "Crafted stone sword."
            r = self._try_unlock("make_stone_sword")
            return r + self._craft_increment()
        return 0.0

    def _handle_make_iron_sword(self) -> float:
        if (
            self._near_table()
            and self._near_furnace()
            and self._inventory.get("wood", 0) >= 1
            and self._inventory.get("iron", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["iron"] -= 1
            self._inventory["iron_sword"] = (
                self._inventory.get("iron_sword", 0) + 1
            )
            self._message = "Crafted iron sword."
            r = self._try_unlock("make_iron_sword")
            return r + self._craft_increment()
        return 0.0

    def _handle_make_diamond_sword(self) -> float:
        if (
            self._near_table()
            and self._near_furnace()
            and self._inventory.get("wood", 0) >= 1
            and self._inventory.get("diamond", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["diamond"] -= 1
            self._inventory["diamond_sword"] = (
                self._inventory.get("diamond_sword", 0) + 1
            )
            self._message = "Crafted diamond sword."
            r = self._try_unlock("make_diamond_sword")
            return r + self._craft_increment()
        return 0.0

    # -- Crafting: armor --

    def _handle_make_wood_armor(self) -> float:
        if (
            self._near_table()
            and self._inventory.get("wood", 0) >= 2
        ):
            self._inventory["wood"] -= 2
            self._inventory["wood_armor"] = (
                self._inventory.get("wood_armor", 0) + 1
            )
            self._message = "Crafted wood armor."
            r = self._try_unlock("make_wood_armor")
            return r + self._craft_increment()
        return 0.0

    def _handle_make_stone_armor(self) -> float:
        # Stone tier needs a table only.
        if (
            self._near_table()
            and self._inventory.get("stone", 0) >= 2
        ):
            self._inventory["stone"] -= 2
            self._inventory["stone_armor"] = (
                self._inventory.get("stone_armor", 0) + 1
            )
            self._message = "Crafted stone armor."
            r = self._try_unlock("make_stone_armor")
            return r + self._craft_increment()
        return 0.0

    def _handle_make_arrow(self) -> float:
        if (
            self._near_table()
            and self._inventory.get("wood", 0) >= 1
            and self._inventory.get("stone", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["stone"] -= 1
            self._inventory["arrows"] = (
                self._inventory.get("arrows", 0) + 2
            )
            self._message = "Crafted 2 arrows."
            return self._try_unlock("make_arrow")
        return 0.0

    def _handle_make_torch(self) -> float:
        if (
            self._near_table()
            and self._inventory.get("wood", 0) >= 1
            and self._inventory.get("coal", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["coal"] -= 1
            self._inventory["torch"] = (
                self._inventory.get("torch", 0) + 4
            )
            self._message = "Crafted 4 torches."
            return self._try_unlock("make_torch")
        return 0.0

    def _handle_make_iron_armor(self) -> float:
        if (
            self._near_table()
            and self._near_furnace()
            and self._inventory.get("iron", 0) >= 2
        ):
            self._inventory["iron"] -= 2
            self._inventory["iron_armor"] = (
                self._inventory.get("iron_armor", 0) + 1
            )
            self._message = "Crafted iron armor."
            r = self._try_unlock("make_iron_armor")
            return r + self._craft_increment()
        return 0.0

    def _handle_make_diamond_armor(self) -> float:
        if (
            self._near_table()
            and self._near_furnace()
            and self._inventory.get("diamond", 0) >= 1
            and self._inventory.get("iron", 0) >= 1
        ):
            self._inventory["diamond"] -= 1
            self._inventory["iron"] -= 1
            self._inventory["diamond_armor"] = (
                self._inventory.get("diamond_armor", 0) + 1
            )
            self._message = "Crafted diamond armor."
            r = self._try_unlock("make_diamond_armor")
            return r + self._craft_increment()
        return 0.0

    # -- Magic --

    def _handle_cast_fireball(self) -> float:
        """Spawn a fireball projectile in front of the agent.

        Mirrors upstream cast_spell:2547-2599 — point projectile, NOT AOE.
        Cost: 2 mana (matches upstream). Phase α uses scalar damage; phase γ
        promotes to a (physical, fire, ice) 3-vector.
        """
        from glyphbench.envs.craftax.mechanics.projectiles import (
            ProjectileEntity,
            ProjectileType,
        )

        if not self._learned_spells["fireball"]:
            self._message = "No spells learned yet."
            return 0.0
        if self._mana < 2:
            self._message = "Not enough mana! (need 2)"
            return 0.0
        # Spawn at the player's tile (upstream spawn_projectile places at
        # state.player_position; the same-step advance by _move_player_projectile
        # then carries it to player_pos + 1*dir). Player is always in-bounds,
        # so the old bounds check on spawn_x/spawn_y was vacuous.
        dx, dy = self._facing
        self._mana -= 2
        self._player_projectiles.append(
            ProjectileEntity(
                kind=ProjectileType.FIREBALL,
                x=self._agent_x, y=self._agent_y, dx=dx, dy=dy,
                damage=4,  # phase-α scalar; phase γ promotes to 3-vector
            )
        )
        self._message = "You cast a fireball."
        return self._try_unlock("cast_fireball")

    def _handle_cast_iceball(self) -> float:
        """Spawn an iceball projectile in front of the agent.

        Mirrors upstream cast_spell:2547-2599 — point projectile, NOT a freeze
        and NOT an AOE. Cost: 2 mana. Phase α uses scalar damage; phase γ
        promotes to a (physical, fire, ice) 3-vector that bypasses ice
        defense.
        """
        from glyphbench.envs.craftax.mechanics.projectiles import (
            ProjectileEntity,
            ProjectileType,
        )

        if not self._learned_spells["iceball"]:
            self._message = "No spells learned yet."
            return 0.0
        if self._mana < 2:
            self._message = "Not enough mana! (need 2)"
            return 0.0
        # Spawn at the player's tile (upstream-faithful; same-step advance
        # carries it to player_pos + 1*dir). Player is always in-bounds.
        dx, dy = self._facing
        self._mana -= 2
        self._player_projectiles.append(
            ProjectileEntity(
                kind=ProjectileType.ICEBALL,
                x=self._agent_x, y=self._agent_y, dx=dx, dy=dy,
                damage=3,
            )
        )
        self._message = "You cast an iceball."
        return self._try_unlock("cast_iceball")

    def _handle_read_book(self) -> float:
        """Read a book to learn an unlearned spell (T11β).

        Consumes 1 book from inventory. Picks a random unlearned spell from
        _learned_spells and marks it True. Fires the learn_<spell> achievement.
        No-ops if the player has no book or already knows all spells.
        """
        if self._inventory.get("book", 0) < 1:
            self._message = "No book to read."
            return 0.0
        unlearned = [
            spell for spell, known in self._learned_spells.items() if not known
        ]
        if not unlearned:
            self._message = "You already know all spells."
            return 0.0
        chosen = unlearned[int(self.rng.integers(0, len(unlearned)))]
        self._learned_spells[chosen] = True
        self._inventory["book"] -= 1
        self._message = f"You read the book. Learned {chosen}!"
        return self._try_unlock(f"learn_{chosen}")

    def _handle_shoot_arrow(self) -> float:
        """Spawn an arrow projectile at the player's tile.

        Mirrors upstream SHOOT_ARROW (action 24). Requires a bow and at least
        1 arrow in inventory. Spawn position is the player's own tile; the
        per-step driver advances the projectile +1 tile/turn in the agent's
        facing direction.
        """
        from glyphbench.envs.craftax.mechanics.projectiles import (
            ProjectileEntity,
            ProjectileType,
        )

        if self._inventory.get("bow", 0) < 1:
            self._message = "You need a bow to shoot arrows."
            return 0.0
        if self._inventory.get("arrows", 0) < 1:
            self._message = "No arrows."
            return 0.0
        dx, dy = self._facing
        self._inventory["arrows"] -= 1
        self._player_projectiles.append(
            ProjectileEntity(
                kind=ProjectileType.ARROW,
                x=self._agent_x, y=self._agent_y,
                dx=dx, dy=dy,
                damage=2,  # phase-α scalar; phase γ promotes to 3-vector
            )
        )
        self._message = "You fired an arrow."
        return self._try_unlock("fire_bow")

    def _attack_mob_kill(self, mob: Mob) -> float:
        """Handle mob death from spells (mob already at <= 0 HP)."""
        if mob in self._mobs:
            self._mobs.remove(mob)
        self._total_kills += 1
        reward = self._check_kill_milestones()
        mtype = mob["type"]
        if mob["is_boss"]:
            fl = mob["floor"]
            self._bosses_alive[fl] = False
            bdef = _BOSS_DEFS.get(fl)
            if bdef:
                reward += self._try_unlock(bdef["ach"])
                loot = f"boss_loot_{fl}"
                self._inventory[loot] = (
                    self._inventory.get(loot, 0) + 1
                )
                reward += self._try_unlock(
                    f"collect_boss_loot_{fl}"
                )
        elif mtype == "zombie":
            reward += self._try_unlock("defeat_zombie")
        elif mtype == "skeleton":
            reward += self._try_unlock("defeat_skeleton")
        elif mtype == "cow":
            self._food = min(_MAX_FOOD, self._food + 5)
            reward += self._try_unlock("eat_cow")
        elif mtype == "kobold":
            reward += self._try_unlock("defeat_kobold")
        elif mtype == "bat":
            reward += self._try_unlock("defeat_bat")
        return reward

    # -- Potions (phase β: 6 color-keyed handlers) --

    def _handle_drink_potion_color(self, color: str, color_idx: int) -> float:
        """Shared implementation for all DRINK_POTION_* handlers.

        *color* is lowercase (e.g. "red"), *color_idx* is the index into
        POTION_COLORS (0=RED, 1=GREEN, …).  The actual effect is determined
        by looking up self._potion_mapping[color_idx] into POTION_EFFECTS —
        this mapping is hidden and never exposed in the observation.
        """
        from glyphbench.envs.craftax.mechanics.potions import (
            POTION_EFFECTS, apply_potion_effect,
        )
        potions_dict = self._inventory.get("potions", {})
        if potions_dict.get(color, 0) < 1:
            self._message = f"No {color} potion."
            return 0.0
        potions_dict[color] -= 1
        effect_idx = self._potion_mapping[color_idx]
        effect = POTION_EFFECTS[effect_idx]
        apply_potion_effect(self, effect)
        self._message = f"You drank a {color} potion."
        return self._try_unlock("drink_potion")

    def _handle_drink_potion_red(self) -> float:
        return self._handle_drink_potion_color("red", 0)

    def _handle_drink_potion_green(self) -> float:
        return self._handle_drink_potion_color("green", 1)

    def _handle_drink_potion_blue(self) -> float:
        return self._handle_drink_potion_color("blue", 2)

    def _handle_drink_potion_pink(self) -> float:
        return self._handle_drink_potion_color("pink", 3)

    def _handle_drink_potion_cyan(self) -> float:
        return self._handle_drink_potion_color("cyan", 4)

    def _handle_drink_potion_yellow(self) -> float:
        return self._handle_drink_potion_color("yellow", 5)

    # -- Eat / drink --

    def _handle_eat_plant(self) -> float:
        fsize = self._floor_size()
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        if (
            0 <= fx < fsize
            and 0 <= fy < fsize
            and self._current_grid()[fy][fx] == TILE_RIPE_PLANT
        ):
            if self._current_floor == 0:
                self._floors[0][fy][fx] = TILE_GRASS
            else:
                self._current_grid()[fy][fx] = (
                    TILE_DUNGEON_FLOOR
                )
            self._food = min(_MAX_FOOD, self._food + 3)
            self._total_plants_eaten += 1
            self._message = "Ate a plant. (+3 food)"
            r = self._try_unlock("eat_plant")
            if self._total_plants_eaten >= 5:
                r += self._try_unlock("eat_5_plants")
            return r
        return 0.0

    def _handle_drink_water(self) -> float:
        fsize = self._floor_size()
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        if (
            0 <= fx < fsize
            and 0 <= fy < fsize
            and self._current_grid()[fy][fx] == TILE_WATER
        ):
            self._water = _MAX_WATER
            self._total_water_drunk += 1
            self._message = "Drank water. (water restored)"
            r = self._try_unlock("collect_drink")
            if self._total_water_drunk >= 10:
                r += self._try_unlock("drink_10_water")
            return r
        return 0.0

    # -- Stairs --

    def _handle_descend(self) -> float:
        grid = self._current_grid()
        ax, ay = self._agent_x, self._agent_y
        if grid[ay][ax] != TILE_STAIRS_DOWN:
            self._message = "No stairs here."
            return 0.0

        new_floor = self._current_floor + 1
        if new_floor > _NUM_DUNGEON_FLOORS:
            self._message = "Cannot go deeper."
            return 0.0

        self._current_floor = new_floor
        # Recompute lightmap on floor entry (T15β).
        self._recompute_lightmap(new_floor)
        # Place agent at stairs up of new floor
        up_pos = self._stairs_up_pos.get(new_floor)
        if up_pos:
            self._agent_x, self._agent_y = up_pos
        else:
            self._agent_x = _DUNGEON_SIZE // 2
            self._agent_y = _DUNGEON_SIZE // 2

        reward = 0.0
        if new_floor == 1:
            self._message = "Entered the dungeon!"
            reward += self._try_unlock("enter_dungeon")
        elif new_floor == 2:
            self._message = "Reached floor 2."
            reward += self._try_unlock("reach_floor_2")
        elif new_floor == 3:
            self._message = "Reached floor 3."
            reward += self._try_unlock("reach_floor_3")
        elif new_floor == 4:
            self._message = "Reached floor 4."
            reward += self._try_unlock("reach_floor_4")
        elif new_floor == 5:
            self._message = "Reached floor 5."
            reward += self._try_unlock("reach_floor_5")
        return reward

    def _handle_ascend(self) -> float:
        grid = self._current_grid()
        ax, ay = self._agent_x, self._agent_y
        if grid[ay][ax] != TILE_STAIRS_UP:
            self._message = "No stairs here."
            return 0.0
        if self._current_floor <= 0:
            self._message = "Already on surface."
            return 0.0

        new_floor = self._current_floor - 1
        self._current_floor = new_floor
        # Recompute lightmap on floor entry (T15β).
        self._recompute_lightmap(new_floor)

        reward = 0.0
        if new_floor == 0:
            # Return to surface
            down_pos = self._stairs_down_pos.get(0)
            if down_pos:
                self._agent_x = down_pos[0]
                self._agent_y = down_pos[1]
            else:
                self._agent_x = _SURFACE_SIZE // 2
                self._agent_y = _SURFACE_SIZE // 2
            self._message = "Returned to the surface!"
            reward += self._try_unlock("return_to_surface")
        else:
            down_pos = self._stairs_down_pos.get(new_floor)
            if down_pos:
                self._agent_x = down_pos[0]
                self._agent_y = down_pos[1]
            self._message = f"Ascended to floor {new_floor}."
        return reward

    # -- Enchantments --

    def _handle_enchant_weapon(self) -> float:
        if self._weapon_enchanted:
            self._message = "Weapon already enchanted."
            return 0.0
        if (
            self._near_table()
            and self._near_furnace()
            and self._inventory.get("diamond", 0) >= 1
            and self._inventory.get("coal", 0) >= 1
        ):
            self._inventory["diamond"] -= 1
            self._inventory["coal"] -= 1
            self._weapon_enchanted = True
            self._message = "Enchanted weapon! (+2 damage)"
            return self._try_unlock("enchant_weapon")
        return 0.0

    def _handle_enchant_armor(self) -> float:
        if self._armor_enchanted:
            self._message = "Armor already enchanted."
            return 0.0
        if (
            self._near_table()
            and self._near_furnace()
            and self._inventory.get("diamond", 0) >= 1
            and self._inventory.get("coal", 0) >= 1
        ):
            self._inventory["diamond"] -= 1
            self._inventory["coal"] -= 1
            self._armor_enchanted = True
            self._message = "Enchanted armor! (+1 defense)"
            return self._try_unlock("enchant_armor")
        return 0.0

    # -- REST --

    def _handle_rest(self) -> float:
        """Enter REST state: regen +1 HP/tick until full, starving, or hit."""
        self._is_resting = True
        self._message = "You rest."
        return 0.0

    # -- Action dispatch table --

    _ACTION_DISPATCH: dict[str, Any] = {  # method refs
        "MOVE_LEFT": _handle_move_left,
        "MOVE_RIGHT": _handle_move_right,
        "MOVE_UP": _handle_move_up,
        "MOVE_DOWN": _handle_move_down,
        "DO": _handle_do,
        "SLEEP": _handle_sleep,
        "PLACE_STONE": _handle_place_stone,
        "PLACE_TABLE": _handle_place_table,
        "PLACE_FURNACE": _handle_place_furnace,
        "PLACE_PLANT": _handle_place_plant,
        "PLACE_TORCH": _handle_place_torch,
        "MAKE_WOOD_PICKAXE": _handle_make_wood_pickaxe,
        "MAKE_STONE_PICKAXE": _handle_make_stone_pickaxe,
        "MAKE_IRON_PICKAXE": _handle_make_iron_pickaxe,
        "MAKE_DIAMOND_PICKAXE": _handle_make_diamond_pickaxe,
        "MAKE_WOOD_SWORD": _handle_make_wood_sword,
        "MAKE_STONE_SWORD": _handle_make_stone_sword,
        "MAKE_IRON_SWORD": _handle_make_iron_sword,
        "MAKE_DIAMOND_SWORD": _handle_make_diamond_sword,
        "MAKE_WOOD_ARMOR": _handle_make_wood_armor,
        "MAKE_STONE_ARMOR": _handle_make_stone_armor,
        "MAKE_IRON_ARMOR": _handle_make_iron_armor,
        "MAKE_DIAMOND_ARMOR": _handle_make_diamond_armor,
        "CAST_FIREBALL": _handle_cast_fireball,
        "CAST_ICEBALL": _handle_cast_iceball,
        "DRINK_POTION_RED": _handle_drink_potion_red,
        "DRINK_POTION_GREEN": _handle_drink_potion_green,
        "DRINK_POTION_BLUE": _handle_drink_potion_blue,
        "DRINK_POTION_PINK": _handle_drink_potion_pink,
        "DRINK_POTION_CYAN": _handle_drink_potion_cyan,
        "DRINK_POTION_YELLOW": _handle_drink_potion_yellow,
        "EAT_PLANT": _handle_eat_plant,
        "DRINK_WATER": _handle_drink_water,
        "DESCEND": _handle_descend,
        "ASCEND": _handle_ascend,
        "ENCHANT_WEAPON": _handle_enchant_weapon,
        "ENCHANT_ARMOR": _handle_enchant_armor,
        "MAKE_ARROW": _handle_make_arrow,
        "MAKE_TORCH": _handle_make_torch,
        "SHOOT_ARROW": _handle_shoot_arrow,
        "REST": _handle_rest,
        "READ_BOOK": _handle_read_book,
    }

    # ---------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------

    def _is_visible(self, wx: int, wy: int) -> bool:
        """Check tile visibility via per-tile lightmap (phase β T16β).

        A tile is visible iff its lightmap value > VISIBILITY_THRESHOLD (0.05).
        Falls back to True if the lightmap has not yet been computed for this floor.
        """
        from glyphbench.envs.craftax.mechanics.lighting import VISIBILITY_THRESHOLD
        lm = self._lightmap.get(self._current_floor)
        if lm is None:
            return True  # fallback: lightmap not yet computed
        if 0 <= wy < lm.shape[0] and 0 <= wx < lm.shape[1]:
            return float(lm[wy, wx]) > VISIBILITY_THRESHOLD
        return False

    def _render_current_observation(self) -> GridObservation:
        half_w = FULL_VIEW_WIDTH // 2
        half_h = FULL_VIEW_HEIGHT // 2
        grid: list[list[str]] = []
        symbols_seen: set[str] = set()
        fsize = self._floor_size()
        cur_grid = self._current_grid()

        # Directional player char
        facing_ch = _DIR_CHARS.get(self._facing, "@")
        facing_name = _DIR_NAMES.get(self._facing, "right")

        mob_chars: dict[tuple[int, int], str] = {}
        visible_mobs: list[Mob] = []
        for mob in self._mobs:
            if mob["floor"] == self._current_floor:
                if mob["is_boss"]:
                    mob_chars[(mob["x"], mob["y"])] = (
                        TILE_BOSS
                    )
                elif mob["type"] in _MOB_TILES:
                    mob_chars[(mob["x"], mob["y"])] = (
                        _MOB_TILES[mob["type"]]
                    )

        from glyphbench.envs.craftax.mechanics.projectiles import ProjectileType
        _PROJECTILE_GLYPH = {
            ProjectileType.ARROW: TILE_ARROW,
            ProjectileType.ARROW2: TILE_ARROW2,
            ProjectileType.DAGGER: TILE_DAGGER,
            ProjectileType.FIREBALL: TILE_FIREBALL,
            ProjectileType.FIREBALL2: TILE_FIREBALL2,
            ProjectileType.ICEBALL: TILE_ICEBALL,
            ProjectileType.ICEBALL2: TILE_ICEBALL2,
            ProjectileType.SLIMEBALL: TILE_SLIMEBALL,
        }
        proj_chars: dict[tuple[int, int], str] = {}
        for p in self._player_projectiles + self._mob_projectiles:
            proj_chars[(p.x, p.y)] = _PROJECTILE_GLYPH[p.kind]

        for wy in range(FULL_VIEW_HEIGHT):
            row: list[str] = []
            for wx in range(FULL_VIEW_WIDTH):
                world_x = self._agent_x - half_w + wx
                world_y = self._agent_y - half_h + wy
                if (
                    world_x == self._agent_x
                    and world_y == self._agent_y
                ):
                    row.append(facing_ch)
                    symbols_seen.add(facing_ch)
                elif (world_x, world_y) in mob_chars:
                    if self._is_visible(world_x, world_y):
                        c = mob_chars[(world_x, world_y)]
                        row.append(c)
                        symbols_seen.add(c)
                    else:
                        row.append(" ")
                elif (world_x, world_y) in proj_chars:
                    if self._is_visible(world_x, world_y):
                        c = proj_chars[(world_x, world_y)]
                        row.append(c)
                        symbols_seen.add(c)
                    else:
                        row.append(" ")
                elif (
                    0 <= world_x < fsize
                    and 0 <= world_y < fsize
                ):
                    if self._is_visible(world_x, world_y):
                        tile = cur_grid[world_y][world_x]
                        row.append(tile)
                        symbols_seen.add(tile)
                    else:
                        row.append(" ")
                else:
                    row.append(" ")
            grid.append(row)

        # Collect visible mobs for HUD
        for mob in self._mobs:
            if mob["floor"] != self._current_floor:
                continue
            dx = mob["x"] - self._agent_x
            dy = mob["y"] - self._agent_y
            if abs(dx) <= half_w and abs(dy) <= half_h:
                visible_mobs.append(mob)

        # Inventory string (skip nested dicts like "potions"; those are shown separately)
        inv_parts: list[str] = []
        for item, count in sorted(self._inventory.items()):
            if isinstance(count, dict):
                continue  # "potions" sub-dict rendered separately
            if count > 0:
                inv_parts.append(f"{item} x{count}")
        inv_str = (
            ", ".join(inv_parts) if inv_parts else "(empty)"
        )

        # Best weapon/armor with numeric values
        best_wpn = "none"
        best_wpn_bonus = 0
        for w in reversed(list(_WEAPON_BONUS.keys())):
            if self._inventory.get(w, 0) > 0:
                best_wpn = w.replace("_", " ")
                best_wpn_bonus = _WEAPON_BONUS[w]
                break
        if best_wpn != "none":
            wpn_str = f"{best_wpn} (+{best_wpn_bonus} dmg)"
            if self._weapon_enchanted:
                wpn_str += " [enchanted +2]"
        else:
            wpn_str = "none"

        best_arm = "none"
        best_arm_def = 0
        for a in reversed(list(_ARMOR_DEFENSE.keys())):
            if self._inventory.get(a, 0) > 0:
                best_arm = a.replace("_", " ")
                best_arm_def = _ARMOR_DEFENSE[a]
                break
        if best_arm != "none":
            arm_str = f"{best_arm} (def {best_arm_def})"
            if self._armor_enchanted:
                arm_str += " [enchanted +1]"
        else:
            arm_str = "none"

        floor_str = (
            "Surface" if self._current_floor == 0
            else f"Dungeon F{self._current_floor}"
        )

        ach_count = len(self._achievements_unlocked)
        total_ach = len(self._ALL_ACHIEVEMENTS)
        # Phase β: per-color potion inventory (never reveal effect mapping)
        potions_dict = self._inventory.get("potions", {})
        potion_parts = [
            f"{color} x{cnt}"
            for color, cnt in potions_dict.items()
            if cnt > 0
        ]
        potions_str = ", ".join(potion_parts) if potion_parts else "none"

        # Day/night cycle position
        cycle_pos = self._day_counter % _CYCLE_LENGTH
        if self._day_night == "day":
            phase_step = cycle_pos
            phase_len = _DAY_LENGTH
        else:
            phase_step = cycle_pos - _DAY_LENGTH
            phase_len = _NIGHT_LENGTH
        time_str = (
            f"{self._day_night} "
            f"(step {phase_step}/{phase_len})"
        )

        # Survival drain countdowns
        food_drain = (
            _FOOD_DRAIN_INTERVAL
            - (self._day_counter % _FOOD_DRAIN_INTERVAL)
        )
        water_drain = (
            _WATER_DRAIN_INTERVAL
            - (self._day_counter % _WATER_DRAIN_INTERVAL)
        )
        energy_drain = (
            _ENERGY_DRAIN_INTERVAL
            - (self._day_counter % _ENERGY_DRAIN_INTERVAL)
        )

        # Nearby mobs info
        mob_parts: list[str] = []
        for m in visible_mobs:
            mob_parts.append(
                f"{m['type']} ({m['hp']}/{m['max_hp']} HP)"
            )
        mob_str = (
            ", ".join(mob_parts) if mob_parts else "none"
        )

        # Boss info for visible bosses
        boss_parts: list[str] = []
        for m in visible_mobs:
            if m["is_boss"]:
                bdef = _BOSS_DEFS.get(m["floor"])
                if bdef:
                    boss_parts.append(
                        f"{bdef['name'].title()} "
                        f"(HP: {m['hp']}/{m['max_hp']}"
                        f", Dmg: {bdef['damage']})"
                    )

        # Spells learned (per-spell dict; T10β)
        spell_parts = [
            f"{spell}: {'known' if known else 'unknown'}"
            for spell, known in self._learned_spells.items()
        ]
        spells_str = ", ".join(spell_parts) if spell_parts else "none"

        # Active potion effects
        effects: list[str] = []
        if self._speed_turns > 0:
            effects.append(
                f"speed ({self._speed_turns} turns)"
            )
        effects_str = (
            ", ".join(effects) if effects else "none"
        )

        # Achievement names
        ach_names = sorted(self._achievements_unlocked)
        ach_name_str = (
            ", ".join(ach_names) if ach_names else "(none)"
        )

        # Growing saplings in viewport
        grow_parts: list[str] = []
        for (px, py), remaining in self._plants.items():
            pdx = px - self._agent_x
            pdy = py - self._agent_y
            if abs(pdx) <= half_w and abs(pdy) <= half_h:
                grow_parts.append(
                    f"sapling at ({px},{py})"
                    f" - {remaining} steps left"
                )

        hud = (
            f"HP: {self._hp}/{self._max_hp}  "
            f"Food: {self._food}/{_MAX_FOOD}  "
            f"Water: {self._water}/{_MAX_WATER}  "
            f"Energy: {self._energy}/{_MAX_ENERGY}  "
            f"Mana: {self._mana}/{_MAX_MANA}\n"
            f"Facing: {facing_name}  "
            f"Floor: {floor_str}  "
            f"Time: {time_str}  "
            f"Step: {self._turn} / {self.max_turns}\n"
            f"Next drain: food in {food_drain}, "
            f"water in {water_drain}, "
            f"energy in {energy_drain}\n"
            f"Weapon: {wpn_str}  Armor: {arm_str}\n"
            f"Spells: {spells_str}\n"
            f"Effects: {effects_str}\n"
            f"Potions: {potions_str}\n"
            f"Nearby mobs: {mob_str}\n"
            f"Inventory: {inv_str}\n"
            f"Achievements: {ach_name_str} "
            f"({ach_count}/{total_ach})"
        )
        if boss_parts:
            hud += "\nBoss: " + "; ".join(boss_parts)
        if grow_parts:
            hud += "\nGrowing: " + "; ".join(grow_parts)

        # Legend -- tile meanings (without player)
        agent_legend = f"you (facing {facing_name})"
        tile_meanings: dict[str, str] = {
            TILE_GRASS: "grass",
            TILE_TREE: "tree (chop for wood)",
            TILE_STONE: "stone (mine with pickaxe)",
            TILE_COAL: "coal ore",
            TILE_IRON: "iron ore",
            TILE_DIAMOND: "diamond",
            TILE_SAPPHIRE: "sapphire ore (needs iron pickaxe)",
            TILE_RUBY: "ruby ore (needs iron pickaxe)",
            TILE_WATER: "water (DRINK_WATER)",
            TILE_LAVA: "lava (2 dmg/tick)",
            TILE_SAND: "sand",
            TILE_TABLE: "crafting table",
            TILE_FURNACE: "furnace",
            TILE_PLACED_STONE: "placed stone",
            TILE_SAPLING: "sapling (growing)",
            TILE_RIPE_PLANT: "ripe plant (EAT_PLANT)",
            TILE_ZOMBIE: "zombie",
            TILE_COW: "cow (passive)",
            TILE_SKELETON_ARCHER: "skeleton (ranged)",  # glyph "a" — upstream ranged skeleton
            TILE_KOBOLD: "kobold (ranged)",
            TILE_BAT: "bat",
            TILE_BOSS: "boss",
            TILE_STAIRS_DOWN: "stairs down (DESCEND)",
            TILE_STAIRS_UP: "stairs up (ASCEND)",
            TILE_TORCH: "torch (light)",
            TILE_DUNGEON_WALL: "dungeon wall",
            TILE_DUNGEON_FLOOR: "dungeon floor",
            TILE_BOSS_DOOR: "boss door",
            # Phase β dungeon features (T18β)
            TILE_CHEST: "chest (DO to open — 1 chest per room)",
            TILE_FOUNTAIN: "fountain (DO to refill water)",
            # Projectiles (phase α — T26)
            TILE_ARROW: "arrow projectile",
            TILE_ARROW2: "arrow2 projectile",
            TILE_DAGGER: "dagger projectile",
            TILE_FIREBALL: "fireball projectile",
            TILE_FIREBALL2: "fireball2 projectile",
            TILE_ICEBALL: "iceball projectile",
            TILE_ICEBALL2: "iceball2 projectile",
            TILE_SLIMEBALL: "slimeball projectile",
        }
        legend_entries: dict[str, str] = {}
        for sym in symbols_seen:
            if sym == facing_ch:
                # Player char; combine with tile meaning if
                # the same symbol is also a tile
                base = tile_meanings.get(sym)
                if base:
                    legend_entries[sym] = (
                        f"{agent_legend} / {base}"
                    )
                else:
                    legend_entries[sym] = agent_legend
            elif sym in tile_meanings:
                legend_entries[sym] = tile_meanings[sym]
        legend = build_legend(legend_entries)

        return GridObservation(
            grid=grid_to_string(grid),
            legend=legend,
            hud=hud,
            message=self._message,
        )
