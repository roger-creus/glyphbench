"""Craftax Classic environment (22 achievements).

A 64x64 grid world with multiple biomes, resource gathering, crafting,
combat, day/night cycles, and survival mechanics (food/water/energy).

Gym ID: atlas_rl/craftax-classic-v0
"""

from __future__ import annotations

from typing import Any, TypedDict

from atlas_rl.core.ascii_primitives import build_legend, grid_to_string
from atlas_rl.core.base_env import BaseAsciiEnv
from atlas_rl.core.observation import GridObservation
from atlas_rl.envs.craftax.base import (
    ALL_CLASSIC_ACHIEVEMENTS,
    CRAFTAX_ACTION_SPEC,
    TILE_AGENT,
    TILE_COAL,
    TILE_COW,
    TILE_DIAMOND,
    TILE_FURNACE,
    TILE_GRASS,
    TILE_IRON,
    TILE_LAVA,
    TILE_PLACED_STONE,
    TILE_PLANT,
    TILE_RIPE_PLANT,
    TILE_SAND,
    TILE_SAPLING,
    TILE_SKELETON,
    TILE_STONE,
    TILE_TABLE,
    TILE_TREE,
    TILE_WATER,
    TILE_ZOMBIE,
    VIEW_HEIGHT,
    VIEW_WIDTH,
)

# Tiles the agent can walk on
WALKABLE_TILES = frozenset({
    TILE_GRASS, TILE_SAND, TILE_TABLE, TILE_FURNACE, TILE_PLACED_STONE,
    TILE_PLANT, TILE_SAPLING, TILE_RIPE_PLANT,
})

# Tiles the agent can interact with via DO (resource tiles only)
INTERACTABLE_TILES: dict[str, str] = {
    TILE_TREE: "wood",
    TILE_STONE: "stone",
    TILE_COAL: "coal",
    TILE_IRON: "iron",
    TILE_DIAMOND: "diamond",
}

# Tiles that require a pickaxe to mine, mapping to minimum required tool
PICKAXE_REQUIRED: dict[str, str] = {
    TILE_STONE: "wood_pickaxe",
    TILE_COAL: "stone_pickaxe",
    TILE_IRON: "stone_pickaxe",
    TILE_DIAMOND: "iron_pickaxe",
}

# Day/night timing (in steps)
_DAY_LENGTH = 200
_NIGHT_LENGTH = 100
_CYCLE_LENGTH = _DAY_LENGTH + _NIGHT_LENGTH

# Survival drain intervals
_FOOD_DRAIN_INTERVAL = 50
_WATER_DRAIN_INTERVAL = 40
_ENERGY_DRAIN_INTERVAL = 100

# Max survival stats
_MAX_FOOD = 9
_MAX_WATER = 9
_MAX_ENERGY = 9

# Mob definitions
_MOB_STATS: dict[str, dict[str, int]] = {
    "zombie": {"hp": 3, "damage": 1},
    "skeleton": {"hp": 4, "damage": 2},
    "cow": {"hp": 3, "damage": 0},
}

_MOB_TILES: dict[str, str] = {
    "zombie": TILE_ZOMBIE,
    "skeleton": TILE_SKELETON,
    "cow": TILE_COW,
}

# Weapon damage bonuses
_WEAPON_BONUS: dict[str, int] = {
    "wood_sword": 1,
    "stone_sword": 2,
    "iron_sword": 3,
}

# Sapling drop chance from chopping trees
_SAPLING_DROP_CHANCE = 0.3

# Steps for sapling to become ripe
_PLANT_RIPEN_STEPS = 20


class Mob(TypedDict):
    """Mob state dictionary."""

    type: str
    x: int
    y: int
    hp: int
    max_hp: int


class CraftaxClassicEnv(BaseAsciiEnv):
    """Craftax Classic: survival crafting in a procedural grid world.

    22 achievements: resource gathering, crafting, combat, survival.
    Features day/night cycles, hostile mobs, passive cows, hunger/thirst/energy.

    World: 64x64 grid. Visible window: 9x7 centered on agent.
    Reward: +1 per first-time achievement unlock.
    """

    action_spec = CRAFTAX_ACTION_SPEC
    noop_action_name = "NOOP"

    _WORLD_SIZE = 64
    _ALL_ACHIEVEMENTS = ALL_CLASSIC_ACHIEVEMENTS

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._world: list[list[str]] = []
        self._agent_x: int = 0
        self._agent_y: int = 0
        self._facing: tuple[int, int] = (1, 0)
        self._inventory: dict[str, int] = {}
        self._achievements_unlocked: set[str] = set()
        self._message: str = ""
        self._hp: int = 9
        self._max_hp: int = 9
        # Survival stats
        self._food: int = _MAX_FOOD
        self._water: int = _MAX_WATER
        self._energy: int = _MAX_ENERGY
        # Day/night
        self._day_counter: int = 0
        self._day_night: str = "day"
        # Mobs
        self._mobs: list[Mob] = []
        # Plants: maps (x, y) -> steps remaining until ripe
        self._plants: dict[tuple[int, int], int] = {}

    def env_id(self) -> str:
        return "atlas_rl/craftax-classic-v0"

    def system_prompt(self) -> str:
        ach_list = ", ".join(self._ALL_ACHIEVEMENTS)
        return (
            "You are playing Craftax Classic.\n\n"
            "TASK\n"
            "Gather resources, craft tools, fight mobs, and survive. "
            "Each new achievement gives +1 reward. Available achievements: "
            f"{ach_list}.\n\n"
            "WORLD\n"
            "64x64 grid world. You see a 9x7 window centered on yourself. "
            "Biomes: grass (.), trees (T), stone (S), coal (C), iron (I), "
            "diamond (D), water (~), lava (L), sand (s).\n"
            "Mobs: zombie (z), skeleton (k), cow (c). "
            "Plants: sapling (;), ripe plant (*).\n\n"
            "SURVIVAL\n"
            "- Food drains 1 every 50 steps. At 0: lose 1 HP per step.\n"
            "- Water drains 1 every 40 steps. At 0: lose 1 HP per step.\n"
            "- Energy drains 1 every 100 steps. At 0: movement costs double.\n"
            "- Eat a ripe plant to restore 3 food. Eat cow meat for 5 food.\n"
            "- DRINK_WATER facing water restores water to 9.\n"
            "- SLEEP restores energy to 9, skips 50 steps.\n\n"
            "DAY/NIGHT\n"
            "Day lasts 200 steps, night lasts 100 steps. Zombies and "
            "skeletons spawn at nightfall and despawn at dawn.\n\n"
            "COMBAT\n"
            "DO facing a mob attacks it. Damage = 1 + weapon bonus "
            "(wood sword +1, stone sword +2, iron sword +3). "
            "Mobs attack when adjacent.\n\n"
            "MECHANICS\n"
            "- MOVE_*: walk in 4 directions on walkable tiles.\n"
            "- DO: chop trees (wood + chance of sapling), mine ores, attack mobs.\n"
            "- PLACE_TABLE: costs 2 wood.\n"
            "- PLACE_FURNACE: costs 4 stone.\n"
            "- PLACE_STONE: costs 1 stone.\n"
            "- PLACE_PLANT: costs 1 sapling. Becomes ripe (*) after 20 steps.\n"
            "- Crafting: MAKE_WOOD_PICKAXE (1 wood, table), "
            "MAKE_STONE_PICKAXE (1 wood + 1 stone, table+furnace), "
            "MAKE_IRON_PICKAXE (1 wood + 1 iron, table+furnace), "
            "MAKE_WOOD_SWORD (1 wood, table), "
            "MAKE_STONE_SWORD (1 wood + 1 stone, table+furnace), "
            "MAKE_IRON_SWORD (1 wood + 1 iron, table+furnace).\n\n"
            "TECH TREE\n"
            "1. Chop trees for wood (+ saplings)\n"
            "2. Place table (2 wood) -> craft wood pickaxe (1 wood)\n"
            "3. Mine stone -> place furnace (4 stone)\n"
            "4. Craft stone pickaxe (1 wood + 1 stone, table+furnace)\n"
            "5. Mine iron/coal with stone pickaxe\n"
            "6. Craft iron pickaxe (1 wood + 1 iron, table+furnace)\n"
            "7. Mine diamond with iron pickaxe\n"
            "8. Craft swords for combat\n"
            "9. Plant saplings, eat ripe plants, drink water, sleep\n\n"
            + self.action_spec.render_for_prompt()
        )

    # -------------------------------------------------------------------
    # World generation
    # -------------------------------------------------------------------

    def _generate_world(self) -> None:
        """Generate a 64x64 world with biomes and resources."""
        size = self._WORLD_SIZE
        self._world = [[TILE_GRASS for _ in range(size)] for _ in range(size)]

        # Place trees (dense forest patches)
        num_forests = int(self.rng.integers(8, 15))
        for _ in range(num_forests):
            cx = int(self.rng.integers(5, size - 5))
            cy = int(self.rng.integers(5, size - 5))
            radius = int(self.rng.integers(3, 7))
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:
                        x, y = cx + dx, cy + dy
                        if 0 <= x < size and 0 <= y < size and self.rng.random() < 0.7:
                            self._world[y][x] = TILE_TREE

        # Place stone deposits
        num_stone = int(self.rng.integers(5, 10))
        for _ in range(num_stone):
            cx = int(self.rng.integers(5, size - 5))
            cy = int(self.rng.integers(5, size - 5))
            radius = int(self.rng.integers(2, 4))
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:
                        x, y = cx + dx, cy + dy
                        if 0 <= x < size and 0 <= y < size and self.rng.random() < 0.6:
                            self._world[y][x] = TILE_STONE

        # Place coal deposits
        num_coal = int(self.rng.integers(3, 6))
        for _ in range(num_coal):
            cx = int(self.rng.integers(10, size - 10))
            cy = int(self.rng.integers(10, size - 10))
            radius = int(self.rng.integers(1, 3))
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:
                        x, y = cx + dx, cy + dy
                        if 0 <= x < size and 0 <= y < size and self.rng.random() < 0.5:
                            self._world[y][x] = TILE_COAL

        # Place iron deposits
        num_iron = int(self.rng.integers(3, 5))
        for _ in range(num_iron):
            cx = int(self.rng.integers(10, size - 10))
            cy = int(self.rng.integers(10, size - 10))
            radius = int(self.rng.integers(1, 3))
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:
                        x, y = cx + dx, cy + dy
                        if 0 <= x < size and 0 <= y < size and self.rng.random() < 0.4:
                            self._world[y][x] = TILE_IRON

        # Place water bodies
        num_water = int(self.rng.integers(2, 5))
        for _ in range(num_water):
            cx = int(self.rng.integers(10, size - 10))
            cy = int(self.rng.integers(10, size - 10))
            radius = int(self.rng.integers(2, 5))
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:
                        x, y = cx + dx, cy + dy
                        if 0 <= x < size and 0 <= y < size and self.rng.random() < 0.8:
                            self._world[y][x] = TILE_WATER

        # Place diamond (rare, single tiles)
        num_diamond = int(self.rng.integers(1, 3))
        for _ in range(num_diamond):
            x = int(self.rng.integers(15, size - 15))
            y = int(self.rng.integers(15, size - 15))
            self._world[y][x] = TILE_DIAMOND

        # Clear a starting area around center
        center = size // 2
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                x, y = center + dx, center + dy
                if 0 <= x < size and 0 <= y < size:
                    self._world[y][x] = TILE_GRASS

    def _spawn_initial_cows(self) -> None:
        """Spawn 3-5 cows in the world at random grass tiles."""
        num_cows = int(self.rng.integers(3, 6))
        size = self._WORLD_SIZE
        for _ in range(num_cows):
            for _attempt in range(50):
                x = int(self.rng.integers(5, size - 5))
                y = int(self.rng.integers(5, size - 5))
                if self._world[y][x] == TILE_GRASS and not self._mob_at(x, y):
                    mob: Mob = {
                        "type": "cow",
                        "x": x,
                        "y": y,
                        "hp": _MOB_STATS["cow"]["hp"],
                        "max_hp": _MOB_STATS["cow"]["hp"],
                    }
                    self._mobs.append(mob)
                    break

    # -------------------------------------------------------------------
    # Adjacency helpers
    # -------------------------------------------------------------------

    def _near_table(self) -> bool:
        """Check if agent is adjacent to a crafting table."""
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx = self._agent_x + dx
            ny = self._agent_y + dy
            if (
                0 <= nx < self._WORLD_SIZE
                and 0 <= ny < self._WORLD_SIZE
                and self._world[ny][nx] == TILE_TABLE
            ):
                return True
        return False

    def _near_furnace(self) -> bool:
        """Check if agent is adjacent to a furnace."""
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx = self._agent_x + dx
            ny = self._agent_y + dy
            if (
                0 <= nx < self._WORLD_SIZE
                and 0 <= ny < self._WORLD_SIZE
                and self._world[ny][nx] == TILE_FURNACE
            ):
                return True
        return False

    # -------------------------------------------------------------------
    # Achievement system
    # -------------------------------------------------------------------

    def _try_unlock_achievement(self, name: str) -> float:
        """Try to unlock an achievement. Returns 1.0 if newly unlocked, else 0.0."""
        if name in self._ALL_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
            self._message = f"ACHIEVEMENT: {name.replace('_', ' ').title()}!"
            return 1.0
        return 0.0

    # -------------------------------------------------------------------
    # Mob helpers
    # -------------------------------------------------------------------

    def _mob_at(self, x: int, y: int) -> Mob | None:
        """Return the mob at (x, y), or None."""
        for mob in self._mobs:
            if mob["x"] == x and mob["y"] == y:
                return mob
        return None

    def _best_weapon_bonus(self) -> int:
        """Return the best weapon damage bonus from inventory."""
        best = 0
        for weapon, bonus in _WEAPON_BONUS.items():
            if self._inventory.get(weapon, 0) > 0 and bonus > best:
                best = bonus
        return best

    def _attack_mob(self, mob: Mob) -> float:
        """Attack a mob. Returns reward from any achievement unlocked."""
        damage = 1 + self._best_weapon_bonus()
        mob["hp"] -= damage
        reward = 0.0
        if mob["hp"] <= 0:
            self._mobs.remove(mob)
            mob_type = mob["type"]
            if mob_type == "zombie":
                self._message = "Defeated a zombie!"
                reward += self._try_unlock_achievement("defeat_zombie")
            elif mob_type == "skeleton":
                self._message = "Defeated a skeleton!"
                reward += self._try_unlock_achievement("defeat_skeleton")
            elif mob_type == "cow":
                # Cow drops food, auto-eat
                self._food = min(_MAX_FOOD, self._food + 5)
                self._message = "Defeated a cow! Ate beef. (+5 food)"
                reward += self._try_unlock_achievement("eat_cow")
        else:
            self._message = f"Hit {mob['type']}! ({mob['hp']}/{mob['max_hp']} HP)"
        return reward

    def _mob_ai(self) -> None:
        """Move mobs and handle mob attacks on the player."""
        for mob in list(self._mobs):
            mx, my = mob["x"], mob["y"]
            mob_type = mob["type"]

            if mob_type == "cow":
                # Passive: random walk
                direction = int(self.rng.integers(0, 5))  # 0=stay, 1-4=move
                ddx, ddy = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)][direction]
            else:
                # Hostile: chase player
                ddx, ddy = 0, 0
                dist_x = self._agent_x - mx
                dist_y = self._agent_y - my
                # Only chase if within 8 tiles
                if abs(dist_x) + abs(dist_y) <= 8:
                    if abs(dist_x) >= abs(dist_y):
                        ddx = 1 if dist_x > 0 else (-1 if dist_x < 0 else 0)
                    else:
                        ddy = 1 if dist_y > 0 else (-1 if dist_y < 0 else 0)

            nx, ny = mx + ddx, my + ddy
            # Don't walk onto player, other mobs, or non-walkable tiles
            if (
                0 <= nx < self._WORLD_SIZE
                and 0 <= ny < self._WORLD_SIZE
                and (nx, ny) != (self._agent_x, self._agent_y)
                and not self._mob_at(nx, ny)
                and self._world[ny][nx] in WALKABLE_TILES
            ):
                mob["x"] = nx
                mob["y"] = ny

            # Attack player if adjacent (hostile mobs only)
            if (
                mob_type != "cow"
                and abs(mob["x"] - self._agent_x) + abs(mob["y"] - self._agent_y) <= 1
            ):
                dmg = _MOB_STATS[mob_type]["damage"]
                self._hp -= dmg
                if not self._message:
                    self._message = f"A {mob_type} hits you for {dmg} damage!"

    def _spawn_night_mobs(self) -> None:
        """Spawn 2-4 hostile mobs near the player at nightfall."""
        num_mobs = int(self.rng.integers(2, 5))
        for _ in range(num_mobs):
            mob_type = "zombie" if self.rng.random() < 0.5 else "skeleton"
            for _attempt in range(20):
                dx = int(self.rng.integers(-6, 7))
                dy = int(self.rng.integers(-6, 7))
                x = self._agent_x + dx
                y = self._agent_y + dy
                dist = abs(dx) + abs(dy)
                if (
                    3 <= dist <= 6
                    and 0 <= x < self._WORLD_SIZE
                    and 0 <= y < self._WORLD_SIZE
                    and self._world[y][x] in WALKABLE_TILES
                    and not self._mob_at(x, y)
                    and (x, y) != (self._agent_x, self._agent_y)
                ):
                    mob: Mob = {
                        "type": mob_type,
                        "x": x,
                        "y": y,
                        "hp": _MOB_STATS[mob_type]["hp"],
                        "max_hp": _MOB_STATS[mob_type]["hp"],
                    }
                    self._mobs.append(mob)
                    break

    def _despawn_night_mobs(self) -> None:
        """Remove all hostile mobs at dawn."""
        self._mobs = [m for m in self._mobs if m["type"] == "cow"]

    # -------------------------------------------------------------------
    # Day/night cycle
    # -------------------------------------------------------------------

    def _advance_day_counter(self, steps: int = 1) -> None:
        """Advance the day/night counter by *steps* and handle transitions."""
        for _ in range(steps):
            old_phase = self._day_night
            self._day_counter += 1
            cycle_pos = self._day_counter % _CYCLE_LENGTH
            new_phase = "day" if cycle_pos < _DAY_LENGTH else "night"
            self._day_night = new_phase

            # Transition events
            if old_phase == "day" and new_phase == "night":
                self._message = "The sun sets. Monsters emerge."
                self._spawn_night_mobs()
            elif old_phase == "night" and new_phase == "day":
                self._message = "The sun rises."
                self._despawn_night_mobs()

    # -------------------------------------------------------------------
    # Survival mechanics
    # -------------------------------------------------------------------

    def _apply_survival_drain(self) -> None:
        """Apply periodic food/water/energy drain. Called each step."""
        step = self._day_counter

        if step > 0 and step % _FOOD_DRAIN_INTERVAL == 0:
            self._food = max(0, self._food - 1)
        if step > 0 and step % _WATER_DRAIN_INTERVAL == 0:
            self._water = max(0, self._water - 1)
        if step > 0 and step % _ENERGY_DRAIN_INTERVAL == 0:
            self._energy = max(0, self._energy - 1)

        # Starvation / dehydration damage
        if self._food == 0:
            self._hp -= 1
        if self._water == 0:
            self._hp -= 1

    # -------------------------------------------------------------------
    # Plant mechanics
    # -------------------------------------------------------------------

    def _tick_plants(self) -> None:
        """Age all planted saplings. Convert ripe ones to TILE_RIPE_PLANT."""
        to_ripen: list[tuple[int, int]] = []
        for pos, remaining in self._plants.items():
            remaining -= 1
            self._plants[pos] = remaining
            if remaining <= 0:
                to_ripen.append(pos)
        for pos in to_ripen:
            del self._plants[pos]
            x, y = pos
            if (
                0 <= x < self._WORLD_SIZE
                and 0 <= y < self._WORLD_SIZE
                and self._world[y][x] == TILE_SAPLING
            ):
                self._world[y][x] = TILE_RIPE_PLANT

    # -------------------------------------------------------------------
    # Shelter check (for sleep safety)
    # -------------------------------------------------------------------

    def _has_shelter(self) -> bool:
        """Check if the player is surrounded by placed stones on all 4 sides."""
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx = self._agent_x + dx
            ny = self._agent_y + dy
            if not (
                0 <= nx < self._WORLD_SIZE
                and 0 <= ny < self._WORLD_SIZE
                and self._world[ny][nx] == TILE_PLACED_STONE
            ):
                return False
        return True

    # -------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:
        self._generate_world()
        self._agent_x = self._WORLD_SIZE // 2
        self._agent_y = self._WORLD_SIZE // 2
        self._facing = (1, 0)
        self._inventory = {}
        self._achievements_unlocked = set()
        self._message = ""
        self._hp = self._max_hp
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._day_counter = 0
        self._day_night = "day"
        self._mobs = []
        self._plants = {}
        self._spawn_initial_cows()
        return self._render_current_observation()

    # -------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        self._message = ""
        reward = 0.0

        if name == "NOOP":
            pass

        elif name in ("MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN"):
            reward += self._handle_move(name)

        elif name == "DO":
            reward += self._handle_do()

        elif name == "SLEEP":
            reward += self._handle_sleep()

        elif name == "PLACE_STONE":
            reward += self._handle_place_stone()

        elif name == "PLACE_TABLE":
            reward += self._handle_place_table()

        elif name == "PLACE_FURNACE":
            reward += self._handle_place_furnace()

        elif name == "PLACE_PLANT":
            reward += self._handle_place_plant()

        elif name == "MAKE_WOOD_PICKAXE":
            reward += self._handle_craft_wood_pickaxe()

        elif name == "MAKE_STONE_PICKAXE":
            reward += self._handle_craft_stone_pickaxe()

        elif name == "MAKE_IRON_PICKAXE":
            reward += self._handle_craft_iron_pickaxe()

        elif name == "MAKE_WOOD_SWORD":
            reward += self._handle_craft_wood_sword()

        elif name == "MAKE_STONE_SWORD":
            reward += self._handle_craft_stone_sword()

        elif name == "MAKE_IRON_SWORD":
            reward += self._handle_craft_iron_sword()

        elif name == "EAT_PLANT":
            reward += self._handle_eat_plant()

        elif name == "DRINK_WATER":
            reward += self._handle_drink_water()

        # --- Post-action ticks ---
        self._advance_day_counter()
        self._apply_survival_drain()
        self._tick_plants()
        self._mob_ai()

        # Check death
        terminated = self._hp <= 0
        if terminated:
            self._message = "You died."

        # Count mobs in sight
        mobs_in_sight = self._count_mobs_in_sight()

        info: dict[str, Any] = {
            "agent_pos": (self._agent_x, self._agent_y),
            "inventory": dict(self._inventory),
            "achievements": list(self._achievements_unlocked),
            "achievements_this_step": (
                [self._message.split(": ")[1].rstrip("!")]
                if self._message.startswith("ACHIEVEMENT")
                else []
            ),
            "biome_at_agent": self._world[self._agent_y][self._agent_x],
            "mobs_in_sight": mobs_in_sight,
            "day_night": self._day_night,
            "food": self._food,
            "water": self._water,
            "energy": self._energy,
        }

        return self._render_current_observation(), reward, terminated, False, info

    # -------------------------------------------------------------------
    # Action handlers
    # -------------------------------------------------------------------

    def _handle_move(self, name: str) -> float:
        """Handle MOVE_LEFT/RIGHT/UP/DOWN."""
        direction_map: dict[str, tuple[int, int]] = {
            "MOVE_LEFT": (-1, 0),
            "MOVE_RIGHT": (1, 0),
            "MOVE_UP": (0, -1),
            "MOVE_DOWN": (0, 1),
        }
        dx, dy = direction_map[name]
        self._facing = (dx, dy)
        nx = self._agent_x + dx
        ny = self._agent_y + dy
        if (
            0 <= nx < self._WORLD_SIZE
            and 0 <= ny < self._WORLD_SIZE
            and self._world[ny][nx] in WALKABLE_TILES
            and not self._mob_at(nx, ny)
        ):
            self._agent_x = nx
            self._agent_y = ny
        return 0.0

    def _handle_do(self) -> float:
        """Handle the DO action: mine resources, chop trees, attack mobs."""
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        if not (0 <= fx < self._WORLD_SIZE and 0 <= fy < self._WORLD_SIZE):
            return 0.0

        reward = 0.0

        # Check for mob at target
        mob = self._mob_at(fx, fy)
        if mob is not None:
            reward += self._attack_mob(mob)
            return reward

        tile = self._world[fy][fx]
        if tile in INTERACTABLE_TILES:
            resource = INTERACTABLE_TILES[tile]
            # Check if pickaxe is required
            if tile in PICKAXE_REQUIRED:
                required_tool = PICKAXE_REQUIRED[tile]
                if self._inventory.get(required_tool, 0) < 1:
                    self._message = (
                        f"You need a {required_tool.replace('_', ' ')} to mine this."
                    )
                    return 0.0
            # Collect the resource
            self._inventory[resource] = self._inventory.get(resource, 0) + 1
            self._world[fy][fx] = TILE_GRASS
            self._message = f"Collected {resource}."
            reward += self._try_unlock_achievement(f"collect_{resource}")

            # Trees may drop saplings
            if tile == TILE_TREE and self.rng.random() < _SAPLING_DROP_CHANCE:
                self._inventory["sapling"] = (
                    self._inventory.get("sapling", 0) + 1
                )
                self._message += " Found a sapling!"
                reward += self._try_unlock_achievement("collect_sapling")

        return reward

    def _handle_sleep(self) -> float:
        """Handle the SLEEP action: restore energy, skip time."""
        reward = 0.0
        self._energy = _MAX_ENERGY

        # Check shelter for safety
        if not self._has_shelter() and self.rng.random() < 0.5:
            # Mob attack during sleep
            damage = int(self.rng.integers(1, 4))
            self._hp -= damage
            self._message = f"Attacked while sleeping! Lost {damage} HP."
        else:
            self._message = "You sleep and wake refreshed."

        # Skip 50 steps worth of day/night
        self._advance_day_counter(50)

        reward += self._try_unlock_achievement("wake_up")
        return reward

    def _handle_place_stone(self) -> float:
        """Handle PLACE_STONE action."""
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        if (
            0 <= fx < self._WORLD_SIZE
            and 0 <= fy < self._WORLD_SIZE
            and self._world[fy][fx] == TILE_GRASS
            and not self._mob_at(fx, fy)
            and self._inventory.get("stone", 0) >= 1
        ):
            self._inventory["stone"] -= 1
            self._world[fy][fx] = TILE_PLACED_STONE
            self._message = "Placed stone."
            return self._try_unlock_achievement("place_stone")
        return 0.0

    def _handle_place_table(self) -> float:
        """Handle PLACE_TABLE action."""
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        if (
            0 <= fx < self._WORLD_SIZE
            and 0 <= fy < self._WORLD_SIZE
            and self._world[fy][fx] == TILE_GRASS
            and not self._mob_at(fx, fy)
            and self._inventory.get("wood", 0) >= 2
        ):
            self._inventory["wood"] -= 2
            self._world[fy][fx] = TILE_TABLE
            self._message = "Placed crafting table."
            return self._try_unlock_achievement("place_table")
        return 0.0

    def _handle_place_furnace(self) -> float:
        """Handle PLACE_FURNACE action."""
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        if (
            0 <= fx < self._WORLD_SIZE
            and 0 <= fy < self._WORLD_SIZE
            and self._world[fy][fx] == TILE_GRASS
            and not self._mob_at(fx, fy)
            and self._inventory.get("stone", 0) >= 4
        ):
            self._inventory["stone"] -= 4
            self._world[fy][fx] = TILE_FURNACE
            self._message = "Placed furnace."
            return self._try_unlock_achievement("place_furnace")
        return 0.0

    def _handle_place_plant(self) -> float:
        """Handle PLACE_PLANT action: place a sapling."""
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        if (
            0 <= fx < self._WORLD_SIZE
            and 0 <= fy < self._WORLD_SIZE
            and self._world[fy][fx] == TILE_GRASS
            and not self._mob_at(fx, fy)
            and self._inventory.get("sapling", 0) >= 1
        ):
            self._inventory["sapling"] -= 1
            self._world[fy][fx] = TILE_SAPLING
            self._plants[(fx, fy)] = _PLANT_RIPEN_STEPS
            self._message = "Planted a sapling."
            return self._try_unlock_achievement("place_plant")
        return 0.0

    def _handle_eat_plant(self) -> float:
        """Handle EAT_PLANT action: eat a ripe plant to restore food."""
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        if (
            0 <= fx < self._WORLD_SIZE
            and 0 <= fy < self._WORLD_SIZE
            and self._world[fy][fx] == TILE_RIPE_PLANT
        ):
            self._world[fy][fx] = TILE_GRASS
            self._food = min(_MAX_FOOD, self._food + 3)
            self._message = "Ate a plant. (+3 food)"
            return self._try_unlock_achievement("eat_plant")
        return 0.0

    def _handle_drink_water(self) -> float:
        """Handle DRINK_WATER action: face water tile to restore water."""
        fx = self._agent_x + self._facing[0]
        fy = self._agent_y + self._facing[1]
        if (
            0 <= fx < self._WORLD_SIZE
            and 0 <= fy < self._WORLD_SIZE
            and self._world[fy][fx] == TILE_WATER
        ):
            self._water = _MAX_WATER
            self._message = "Drank water. (water restored)"
            return self._try_unlock_achievement("collect_drink")
        return 0.0

    def _handle_craft_wood_pickaxe(self) -> float:
        """Handle MAKE_WOOD_PICKAXE action."""
        if self._near_table() and self._inventory.get("wood", 0) >= 1:
            self._inventory["wood"] -= 1
            self._inventory["wood_pickaxe"] = (
                self._inventory.get("wood_pickaxe", 0) + 1
            )
            self._message = "Crafted wood pickaxe."
            return self._try_unlock_achievement("make_wood_pickaxe")
        return 0.0

    def _handle_craft_stone_pickaxe(self) -> float:
        """Handle MAKE_STONE_PICKAXE action."""
        if (
            self._near_table()
            and self._near_furnace()
            and self._inventory.get("wood", 0) >= 1
            and self._inventory.get("stone", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["stone"] -= 1
            self._inventory["stone_pickaxe"] = (
                self._inventory.get("stone_pickaxe", 0) + 1
            )
            self._message = "Crafted stone pickaxe."
            return self._try_unlock_achievement("make_stone_pickaxe")
        return 0.0

    def _handle_craft_iron_pickaxe(self) -> float:
        """Handle MAKE_IRON_PICKAXE action."""
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
            return self._try_unlock_achievement("make_iron_pickaxe")
        return 0.0

    def _handle_craft_wood_sword(self) -> float:
        """Handle MAKE_WOOD_SWORD action."""
        if self._near_table() and self._inventory.get("wood", 0) >= 1:
            self._inventory["wood"] -= 1
            self._inventory["wood_sword"] = (
                self._inventory.get("wood_sword", 0) + 1
            )
            self._message = "Crafted wood sword."
            return self._try_unlock_achievement("make_wood_sword")
        return 0.0

    def _handle_craft_stone_sword(self) -> float:
        """Handle MAKE_STONE_SWORD action."""
        if (
            self._near_table()
            and self._near_furnace()
            and self._inventory.get("wood", 0) >= 1
            and self._inventory.get("stone", 0) >= 1
        ):
            self._inventory["wood"] -= 1
            self._inventory["stone"] -= 1
            self._inventory["stone_sword"] = (
                self._inventory.get("stone_sword", 0) + 1
            )
            self._message = "Crafted stone sword."
            return self._try_unlock_achievement("make_stone_sword")
        return 0.0

    def _handle_craft_iron_sword(self) -> float:
        """Handle MAKE_IRON_SWORD action."""
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
            return self._try_unlock_achievement("make_iron_sword")
        return 0.0

    # -------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------

    def _count_mobs_in_sight(self) -> int:
        """Count mobs within the visible window."""
        half_w = VIEW_WIDTH // 2
        half_h = VIEW_HEIGHT // 2
        count = 0
        for mob in self._mobs:
            if (
                abs(mob["x"] - self._agent_x) <= half_w
                and abs(mob["y"] - self._agent_y) <= half_h
            ):
                count += 1
        return count

    def _render_current_observation(self) -> GridObservation:
        """Build the 9x7 visible window centered on the agent."""
        half_w = VIEW_WIDTH // 2
        half_h = VIEW_HEIGHT // 2
        grid: list[list[str]] = []
        symbols_seen: set[str] = set()

        # Pre-compute mob positions in visible range
        mob_chars: dict[tuple[int, int], str] = {}
        for mob in self._mobs:
            mob_chars[(mob["x"], mob["y"])] = _MOB_TILES[mob["type"]]

        for wy in range(VIEW_HEIGHT):
            row: list[str] = []
            for wx in range(VIEW_WIDTH):
                world_x = self._agent_x - half_w + wx
                world_y = self._agent_y - half_h + wy
                if world_x == self._agent_x and world_y == self._agent_y:
                    row.append(TILE_AGENT)
                    symbols_seen.add(TILE_AGENT)
                elif (world_x, world_y) in mob_chars:
                    char = mob_chars[(world_x, world_y)]
                    row.append(char)
                    symbols_seen.add(char)
                elif (
                    0 <= world_x < self._WORLD_SIZE
                    and 0 <= world_y < self._WORLD_SIZE
                ):
                    tile = self._world[world_y][world_x]
                    row.append(tile)
                    symbols_seen.add(tile)
                else:
                    row.append(" ")
            grid.append(row)

        # Build inventory string
        inv_parts: list[str] = []
        for item, count in sorted(self._inventory.items()):
            if count > 0:
                inv_parts.append(f"{item} x{count}")
        inv_str = ", ".join(inv_parts) if inv_parts else "(empty)"

        # Achievements
        ach_names = sorted(self._achievements_unlocked)
        ach_str = ", ".join(ach_names) if ach_names else "(none)"
        ach_count = len(self._achievements_unlocked)

        hud = (
            f"HP: {self._hp}/{self._max_hp}   "
            f"Food: {self._food}/{_MAX_FOOD}   "
            f"Water: {self._water}/{_MAX_WATER}   "
            f"Energy: {self._energy}/{_MAX_ENERGY}   "
            f"Time: {self._day_night}   "
            f"Step: {self._turn}\n"
            f"Inventory: {inv_str}\n"
            f"Achievements: {ach_str} ({ach_count}/{len(self._ALL_ACHIEVEMENTS)})"
        )

        # Build legend from symbols actually seen
        symbol_meanings: dict[str, str] = {
            TILE_AGENT: "you",
            TILE_GRASS: "grass",
            TILE_TREE: "tree (chop with DO for wood)",
            TILE_STONE: "stone (mine with wood pickaxe)",
            TILE_COAL: "coal ore (mine with stone pickaxe)",
            TILE_IRON: "iron ore (mine with stone pickaxe)",
            TILE_DIAMOND: "diamond (mine with iron pickaxe)",
            TILE_WATER: "water (impassable, DRINK_WATER to drink)",
            TILE_LAVA: "lava (impassable, deadly)",
            TILE_SAND: "sand",
            TILE_TABLE: "crafting table",
            TILE_FURNACE: "furnace",
            TILE_PLACED_STONE: "placed stone",
            TILE_PLANT: "plant",
            TILE_SAPLING: "sapling (growing)",
            TILE_RIPE_PLANT: "ripe plant (EAT_PLANT to eat)",
            TILE_ZOMBIE: "zombie (hostile)",
            TILE_SKELETON: "skeleton (hostile)",
            TILE_COW: "cow (passive, drops food)",
        }
        legend_entries: dict[str, str] = {}
        for sym in symbols_seen:
            if sym in symbol_meanings:
                legend_entries[sym] = symbol_meanings[sym]
        legend = build_legend(legend_entries)

        return GridObservation(
            grid=grid_to_string(grid),
            legend=legend,
            hud=hud,
            message=self._message,
        )
