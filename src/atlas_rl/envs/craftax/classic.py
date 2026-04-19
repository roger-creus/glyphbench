"""Craftax Classic environment (Stage 0 subset).

A 64x64 grid world with multiple biomes, resource gathering, and crafting.
Stage 0 implements the first 8 achievements only. No combat, no day/night,
no survival mechanics (food/water/energy drain).

Gym ID: atlas_rl/craftax-classic-v0
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atlas_rl.core.action import ActionSpec
from atlas_rl.core.ascii_primitives import build_legend, grid_to_string
from atlas_rl.core.base_env import BaseAsciiEnv
from atlas_rl.core.observation import GridObservation
from atlas_rl.envs.craftax.base import (
    CRAFTAX_ACTION_SPEC,
    STAGE0_ACHIEVEMENTS,
    TILE_AGENT,
    TILE_COAL,
    TILE_DIAMOND,
    TILE_FURNACE,
    TILE_GRASS,
    TILE_IRON,
    TILE_LAVA,
    TILE_PLACED_STONE,
    TILE_PLANT,
    TILE_SAND,
    TILE_STONE,
    TILE_TABLE,
    TILE_TREE,
    TILE_WATER,
    VIEW_HEIGHT,
    VIEW_WIDTH,
)


# Tiles the agent can walk on
WALKABLE_TILES = frozenset({
    TILE_GRASS, TILE_SAND, TILE_TABLE, TILE_FURNACE, TILE_PLACED_STONE, TILE_PLANT,
})

# Tiles the agent can interact with via DO
INTERACTABLE_TILES: dict[str, str] = {
    TILE_TREE: "wood",
    TILE_STONE: "stone",
    TILE_COAL: "coal",
    TILE_IRON: "iron",
    TILE_DIAMOND: "diamond",
}

# Tiles that require a pickaxe to mine
PICKAXE_REQUIRED: dict[str, str] = {
    TILE_STONE: "wood_pickaxe",
    TILE_COAL: "stone_pickaxe",
    TILE_IRON: "stone_pickaxe",
    TILE_DIAMOND: "stone_pickaxe",  # Would need iron pickaxe in full game
}


class CraftaxClassicEnv(BaseAsciiEnv):
    """Craftax Classic: survival crafting in a procedural grid world.

    Stage 0 subset: 8 achievements (collect wood, place table, make wood
    pickaxe, collect stone, place furnace, make stone pickaxe, collect iron,
    collect coal). No combat, no survival drain, no day/night.

    World: 64x64 grid. Visible window: 9x7 centered on agent.
    Reward: +1 per first-time achievement unlock.
    """

    action_spec = CRAFTAX_ACTION_SPEC
    noop_action_name = "NOOP"

    _WORLD_SIZE = 64
    _STAGE0_ACHIEVEMENTS = STAGE0_ACHIEVEMENTS

    def __init__(self, max_turns: int = 10000) -> None:
        super().__init__(max_turns=max_turns)
        self._world: list[list[str]] = []
        self._agent_x: int = 0
        self._agent_y: int = 0
        self._facing: tuple[int, int] = (1, 0)  # facing right
        self._inventory: dict[str, int] = {}
        self._achievements_unlocked: set[str] = set()
        self._message: str = ""
        self._hp: int = 9
        self._max_hp: int = 9

    def env_id(self) -> str:
        return "atlas_rl/craftax-classic-v0"

    def system_prompt(self) -> str:
        return (
            "You are playing Craftax Classic.\n\n"
            "TASK\n"
            "Gather resources, craft tools, and unlock achievements. "
            "Each new achievement gives +1 reward. Available achievements: "
            "collect_wood, place_table, make_wood_pickaxe, collect_stone, "
            "place_furnace, make_stone_pickaxe, collect_iron, collect_coal.\n\n"
            "WORLD\n"
            "64x64 grid world. You see a 9x7 window centered on yourself. "
            "Biomes: grass (.), trees (T), stone (S), coal (C), iron (I), "
            "diamond (D), water (~), lava (L), sand (s).\n\n"
            "MECHANICS\n"
            "- MOVE_*: walk in 4 directions. You can walk on grass, sand, "
            "tables, furnaces, placed stone, and plants. Trees, stone, ores, "
            "water, and lava block movement.\n"
            "- DO: interact with the block you face. Chop trees for wood, "
            "mine stone/ores with the right pickaxe.\n"
            "- PLACE_TABLE: costs 2 wood. Place a crafting table (t).\n"
            "- PLACE_FURNACE: costs 4 stone. Place a furnace (f).\n"
            "- MAKE_WOOD_PICKAXE: costs 1 wood. Requires adjacent table.\n"
            "- MAKE_STONE_PICKAXE: costs 1 wood + 1 stone. Requires adjacent "
            "table and furnace.\n\n"
            "TECH TREE\n"
            "1. Chop trees (DO) to get wood\n"
            "2. Place a crafting table (PLACE_TABLE, costs 2 wood)\n"
            "3. Make wood pickaxe (MAKE_WOOD_PICKAXE, costs 1 wood, near table)\n"
            "4. Mine stone with wood pickaxe (DO on S)\n"
            "5. Place furnace (PLACE_FURNACE, costs 4 stone)\n"
            "6. Make stone pickaxe (MAKE_STONE_PICKAXE, costs 1 wood + 1 stone, "
            "near table + furnace)\n"
            "7. Mine iron and coal with stone pickaxe (DO on I or C)\n\n"
            + self.action_spec.render_for_prompt()
        )

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
                        if 0 <= x < size and 0 <= y < size:
                            if self.rng.random() < 0.7:
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
                        if 0 <= x < size and 0 <= y < size:
                            if self.rng.random() < 0.6:
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
                        if 0 <= x < size and 0 <= y < size:
                            if self.rng.random() < 0.5:
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
                        if 0 <= x < size and 0 <= y < size:
                            if self.rng.random() < 0.4:
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
                        if 0 <= x < size and 0 <= y < size:
                            if self.rng.random() < 0.8:
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

    def _near_table(self) -> bool:
        """Check if agent is adjacent to a crafting table."""
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx = self._agent_x + dx
            ny = self._agent_y + dy
            if 0 <= nx < self._WORLD_SIZE and 0 <= ny < self._WORLD_SIZE:
                if self._world[ny][nx] == TILE_TABLE:
                    return True
        return False

    def _near_furnace(self) -> bool:
        """Check if agent is adjacent to a furnace."""
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx = self._agent_x + dx
            ny = self._agent_y + dy
            if 0 <= nx < self._WORLD_SIZE and 0 <= ny < self._WORLD_SIZE:
                if self._world[ny][nx] == TILE_FURNACE:
                    return True
        return False

    def _try_unlock_achievement(self, name: str) -> float:
        """Try to unlock an achievement. Returns 1.0 if newly unlocked, 0.0 otherwise."""
        if name in self._STAGE0_ACHIEVEMENTS and name not in self._achievements_unlocked:
            self._achievements_unlocked.add(name)
            self._message = f"ACHIEVEMENT: {name.replace('_', ' ').title()}!"
            return 1.0
        return 0.0

    def _reset(self, seed: int) -> GridObservation:
        self._generate_world()
        self._agent_x = self._WORLD_SIZE // 2
        self._agent_y = self._WORLD_SIZE // 2
        self._facing = (1, 0)  # facing right
        self._inventory = {}
        self._achievements_unlocked = set()
        self._message = ""
        self._hp = self._max_hp
        return self._render_current_observation()

    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        name = self.action_spec.names[action]
        self._message = ""
        reward = 0.0

        if name == "NOOP":
            pass

        elif name == "MOVE_LEFT":
            self._facing = (-1, 0)
            nx = self._agent_x - 1
            if 0 <= nx < self._WORLD_SIZE:
                tile = self._world[self._agent_y][nx]
                if tile in WALKABLE_TILES:
                    self._agent_x = nx

        elif name == "MOVE_RIGHT":
            self._facing = (1, 0)
            nx = self._agent_x + 1
            if 0 <= nx < self._WORLD_SIZE:
                tile = self._world[self._agent_y][nx]
                if tile in WALKABLE_TILES:
                    self._agent_x = nx

        elif name == "MOVE_UP":
            self._facing = (0, -1)
            ny = self._agent_y - 1
            if 0 <= ny < self._WORLD_SIZE:
                tile = self._world[ny][self._agent_x]
                if tile in WALKABLE_TILES:
                    self._agent_y = ny

        elif name == "MOVE_DOWN":
            self._facing = (0, 1)
            ny = self._agent_y + 1
            if 0 <= ny < self._WORLD_SIZE:
                tile = self._world[ny][self._agent_x]
                if tile in WALKABLE_TILES:
                    self._agent_y = ny

        elif name == "DO":
            # Interact with the tile in the facing direction
            fx = self._agent_x + self._facing[0]
            fy = self._agent_y + self._facing[1]
            if 0 <= fx < self._WORLD_SIZE and 0 <= fy < self._WORLD_SIZE:
                tile = self._world[fy][fx]
                if tile in INTERACTABLE_TILES:
                    resource = INTERACTABLE_TILES[tile]
                    # Check if pickaxe is required
                    if tile in PICKAXE_REQUIRED:
                        required_tool = PICKAXE_REQUIRED[tile]
                        if self._inventory.get(required_tool, 0) < 1:
                            self._message = f"You need a {required_tool.replace('_', ' ')} to mine this."
                        else:
                            self._inventory[resource] = self._inventory.get(resource, 0) + 1
                            self._world[fy][fx] = TILE_GRASS
                            self._message = f"Collected {resource}."
                            reward += self._try_unlock_achievement(f"collect_{resource}")
                    else:
                        # No tool required (trees)
                        self._inventory[resource] = self._inventory.get(resource, 0) + 1
                        self._world[fy][fx] = TILE_GRASS
                        self._message = f"Collected {resource}."
                        reward += self._try_unlock_achievement(f"collect_{resource}")

        elif name == "SLEEP":
            # No-op in Stage 0
            pass

        elif name == "PLACE_STONE":
            fx = self._agent_x + self._facing[0]
            fy = self._agent_y + self._facing[1]
            if (0 <= fx < self._WORLD_SIZE and 0 <= fy < self._WORLD_SIZE
                    and self._world[fy][fx] == TILE_GRASS
                    and self._inventory.get("stone", 0) >= 1):
                self._inventory["stone"] -= 1
                self._world[fy][fx] = TILE_PLACED_STONE
                self._message = "Placed stone."

        elif name == "PLACE_TABLE":
            fx = self._agent_x + self._facing[0]
            fy = self._agent_y + self._facing[1]
            if (0 <= fx < self._WORLD_SIZE and 0 <= fy < self._WORLD_SIZE
                    and self._world[fy][fx] == TILE_GRASS
                    and self._inventory.get("wood", 0) >= 2):
                self._inventory["wood"] -= 2
                self._world[fy][fx] = TILE_TABLE
                self._message = "Placed crafting table."
                reward += self._try_unlock_achievement("place_table")

        elif name == "PLACE_FURNACE":
            fx = self._agent_x + self._facing[0]
            fy = self._agent_y + self._facing[1]
            if (0 <= fx < self._WORLD_SIZE and 0 <= fy < self._WORLD_SIZE
                    and self._world[fy][fx] == TILE_GRASS
                    and self._inventory.get("stone", 0) >= 4):
                self._inventory["stone"] -= 4
                self._world[fy][fx] = TILE_FURNACE
                self._message = "Placed furnace."
                reward += self._try_unlock_achievement("place_furnace")

        elif name == "PLACE_PLANT":
            # No-op in Stage 0
            pass

        elif name == "MAKE_WOOD_PICKAXE":
            if (self._near_table()
                    and self._inventory.get("wood", 0) >= 1):
                self._inventory["wood"] -= 1
                self._inventory["wood_pickaxe"] = self._inventory.get("wood_pickaxe", 0) + 1
                self._message = "Crafted wood pickaxe."
                reward += self._try_unlock_achievement("make_wood_pickaxe")

        elif name == "MAKE_STONE_PICKAXE":
            if (self._near_table() and self._near_furnace()
                    and self._inventory.get("wood", 0) >= 1
                    and self._inventory.get("stone", 0) >= 1):
                self._inventory["wood"] -= 1
                self._inventory["stone"] -= 1
                self._inventory["stone_pickaxe"] = self._inventory.get("stone_pickaxe", 0) + 1
                self._message = "Crafted stone pickaxe."
                reward += self._try_unlock_achievement("make_stone_pickaxe")

        elif name == "MAKE_WOOD_SWORD":
            # No-op in Stage 0
            pass

        elif name == "MAKE_STONE_SWORD":
            # No-op in Stage 0
            pass

        elif name == "EAT_PLANT":
            # No-op in Stage 0
            pass

        elif name == "DRINK_WATER":
            # No-op in Stage 0
            pass

        info: dict[str, Any] = {
            "agent_pos": (self._agent_x, self._agent_y),
            "inventory": dict(self._inventory),
            "achievements": list(self._achievements_unlocked),
            "achievements_this_step": [self._message.split(": ")[1].rstrip("!")] if self._message.startswith("ACHIEVEMENT") else [],
            "biome_at_agent": self._world[self._agent_y][self._agent_x],
            "mobs_in_sight": 0,  # No mobs in Stage 0
        }

        return self._render_current_observation(), reward, False, False, info

    def _render_current_observation(self) -> GridObservation:
        # Build 9x7 window centered on agent
        half_w = VIEW_WIDTH // 2
        half_h = VIEW_HEIGHT // 2
        grid: list[list[str]] = []

        # Track which symbols appear for the legend
        symbols_seen: set[str] = set()

        for wy in range(VIEW_HEIGHT):
            row: list[str] = []
            for wx in range(VIEW_WIDTH):
                world_x = self._agent_x - half_w + wx
                world_y = self._agent_y - half_h + wy
                if world_x == self._agent_x and world_y == self._agent_y:
                    row.append(TILE_AGENT)
                    symbols_seen.add(TILE_AGENT)
                elif 0 <= world_x < self._WORLD_SIZE and 0 <= world_y < self._WORLD_SIZE:
                    tile = self._world[world_y][world_x]
                    row.append(tile)
                    symbols_seen.add(tile)
                else:
                    row.append(" ")  # out of bounds
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
            f"Step: {self._turn}\n"
            f"Inventory: {inv_str}\n"
            f"Achievements: {ach_str} ({ach_count}/{len(self._STAGE0_ACHIEVEMENTS)})"
        )

        # Build legend from symbols actually seen
        symbol_meanings: dict[str, str] = {
            TILE_AGENT: "you",
            TILE_GRASS: "grass",
            TILE_TREE: "tree (chop with DO for wood)",
            TILE_STONE: "stone (mine with wood pickaxe)",
            TILE_COAL: "coal ore (mine with stone pickaxe)",
            TILE_IRON: "iron ore (mine with stone pickaxe)",
            TILE_DIAMOND: "diamond (mine with stone pickaxe)",
            TILE_WATER: "water (impassable)",
            TILE_LAVA: "lava (impassable, deadly)",
            TILE_SAND: "sand",
            TILE_TABLE: "crafting table (placed)",
            TILE_FURNACE: "furnace (placed)",
            TILE_PLACED_STONE: "placed stone",
            TILE_PLANT: "plant",
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
