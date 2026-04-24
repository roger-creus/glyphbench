"""Extended Craftax sub-task environments (23 focused tasks).

Each environment subclasses CraftaxClassicEnv or CraftaxFullEnv and
overrides ``_reset`` to set up a curated initial state, and ``_step``
to check sub-task-specific termination and reward.

Gym IDs: glyphbench/craftax-<taskname>-v0
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.observation import GridObservation
from glyphbench.envs.craftax.base import (
    TILE_BAT,
    TILE_BOSS,
    TILE_BOSS_DOOR,
    TILE_COW,
    TILE_DIAMOND,
    TILE_DUNGEON_FLOOR,
    TILE_DUNGEON_WALL,
    TILE_FURNACE,
    TILE_GRASS,
    TILE_IRON,
    TILE_PLACED_STONE,
    TILE_RIPE_PLANT,
    TILE_SAND,
    TILE_SAPLING,
    TILE_SKELETON,
    TILE_SKELETON_ARCHER,
    TILE_SPIDER,
    TILE_STAIRS_DOWN,
    TILE_STAIRS_UP,
    TILE_STONE,
    TILE_TABLE,
    TILE_TORCH,
    TILE_TREE,
    TILE_WATER,
    TILE_ZOMBIE,
)
from glyphbench.envs.craftax.classic import CraftaxClassicEnv
from glyphbench.envs.craftax.classic import Mob as ClassicMob
from glyphbench.envs.craftax.classic import (
    WALKABLE_TILES as CLASSIC_WALKABLE,
    _MOB_STATS as CLASSIC_MOB_STATS,
    _MAX_FOOD,
    _MAX_WATER,
    _MAX_ENERGY,
    _DAY_LENGTH,
    _CYCLE_LENGTH,
)
from glyphbench.envs.craftax.full import CraftaxFullEnv
from glyphbench.envs.craftax.full import Mob as FullMob
from glyphbench.envs.craftax.full import (
    _MOB_STATS as FULL_MOB_STATS,
    _BOSS_DEFS,
    _NUM_DUNGEON_FLOORS,
    _DUNGEON_SIZE,
    _SURFACE_SIZE,
    _MAX_MANA,
    DUNGEON_WALKABLE,
    SURFACE_WALKABLE,
)

# ===================================================================
# Helper: place a Classic mob near a position
# ===================================================================

def _place_classic_mob(
    env: CraftaxClassicEnv,
    mob_type: str,
    near_x: int,
    near_y: int,
    radius: int = 4,
) -> None:
    """Place a mob of *mob_type* near (*near_x*, *near_y*) on walkable grass."""
    stats = CLASSIC_MOB_STATS[mob_type]
    for _att in range(60):
        dx = int(env.rng.integers(-radius, radius + 1))
        dy = int(env.rng.integers(-radius, radius + 1))
        x, y = near_x + dx, near_y + dy
        if (
            0 <= x < env._WORLD_SIZE
            and 0 <= y < env._WORLD_SIZE
            and env._world[y][x] in CLASSIC_WALKABLE
            and (x, y) != (env._agent_x, env._agent_y)
            and not env._mob_at(x, y)
        ):
            mob: ClassicMob = {
                "type": mob_type,
                "x": x,
                "y": y,
                "hp": stats["hp"],
                "max_hp": stats["hp"],
            }
            env._mobs.append(mob)
            return


def _place_full_mob(
    env: CraftaxFullEnv,
    mob_type: str,
    near_x: int,
    near_y: int,
    floor: int = 0,
    radius: int = 4,
    is_boss: bool = False,
    hp: int | None = None,
    max_hp: int | None = None,
) -> None:
    """Place a mob on the Full env's specified floor."""
    stats = FULL_MOB_STATS.get(mob_type, {"hp": 5, "damage": 2})
    actual_hp = hp if hp is not None else stats["hp"]
    actual_max = max_hp if max_hp is not None else stats["hp"]
    fsize = _SURFACE_SIZE if floor == 0 else _DUNGEON_SIZE
    grid = env._floors[floor]
    walkable = SURFACE_WALKABLE if floor == 0 else DUNGEON_WALKABLE
    for _att in range(80):
        dx = int(env.rng.integers(-radius, radius + 1))
        dy = int(env.rng.integers(-radius, radius + 1))
        x, y = near_x + dx, near_y + dy
        if (
            0 <= x < fsize
            and 0 <= y < fsize
            and grid[y][x] in walkable
            and (x, y) != (env._agent_x, env._agent_y)
            and not env._mob_at(x, y, floor)
        ):
            mob: FullMob = {
                "type": mob_type,
                "x": x,
                "y": y,
                "hp": actual_hp,
                "max_hp": actual_max,
                "frozen_turns": 0,
                "is_boss": is_boss,
                "floor": floor,
            }
            env._mobs.append(mob)
            return


def _clear_area(grid: list[list[str]], cx: int, cy: int, r: int,
                size: int, tile: str = TILE_GRASS) -> None:
    """Set all tiles within radius *r* of (*cx*, *cy*) to *tile*."""
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < size and 0 <= y < size:
                grid[y][x] = tile


def _clear_dungeon_area(grid: list[list[str]], cx: int, cy: int, r: int,
                        size: int) -> None:
    """Set dungeon tiles to floor within radius."""
    _clear_area(grid, cx, cy, r, size, tile=TILE_DUNGEON_FLOOR)


# ===================================================================
# 1-4: Per-Floor Dungeon Tasks (Full env)
# ===================================================================

class CraftaxFloor1Env(CraftaxFullEnv):
    """Enter dungeon floor 1. Start with stone sword + stone pickaxe.
    Goal: find and reach the stairs down. Max 100 steps."""

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-floor1-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX DUNGEON FLOOR 1\n\n"
            "You are in the first dungeon floor. Enemies lurk in the dark.\n"
            "GOAL: Find the stairs down and step on them, then DESCEND.\n"
            "You start with a stone sword and stone pickaxe.\n"
            "REWARD: +10 for descending to floor 2.\n"
            "ENEMIES: zombies, skeletons, skeleton archers, spiders, bats.\n"
            "Place torches (PLACE_TORCH) to illuminate dark areas.\n"
            "Attack enemies by facing them and using DO.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        obs = super()._reset(seed)
        # Teleport player to floor 1
        self._current_floor = 1
        up_pos = self._stairs_up_pos.get(1)
        if up_pos:
            self._agent_x, self._agent_y = up_pos
        else:
            self._agent_x = _DUNGEON_SIZE // 2
            self._agent_y = _DUNGEON_SIZE // 2
        # Give starting gear
        self._inventory = {
            "stone_sword": 1,
            "stone_pickaxe": 1,
            "wood": 3,
            "coal": 3,
        }
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_floor = self._current_floor
        obs, reward, terminated, truncated, info = super()._step(action)
        # Check if player descended to floor 2
        if self._current_floor == 2 and old_floor == 1:
            reward += 10.0
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxFloor2Env(CraftaxFullEnv):
    """Dungeon floor 2. Start with iron sword + iron armor.
    Goal: find stairs down. Max 100 steps."""

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-floor2-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX DUNGEON FLOOR 2\n\n"
            "You are on the second dungeon floor. Enemies are tougher.\n"
            "GOAL: Find the stairs down and DESCEND to floor 3.\n"
            "You start with iron sword, iron armor, and torches.\n"
            "REWARD: +10 for descending.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._current_floor = 2
        up_pos = self._stairs_up_pos.get(2)
        if up_pos:
            self._agent_x, self._agent_y = up_pos
        else:
            self._agent_x = _DUNGEON_SIZE // 2
            self._agent_y = _DUNGEON_SIZE // 2
        self._inventory = {
            "iron_sword": 1,
            "iron_armor": 1,
            "wood": 5,
            "coal": 5,
        }
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_floor = self._current_floor
        obs, reward, terminated, truncated, info = super()._step(action)
        if self._current_floor == 3 and old_floor == 2:
            reward += 10.0
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxFloor3Env(CraftaxFullEnv):
    """Dungeon floor 3. Start with iron sword + iron armor + spells.
    Goal: find stairs down. Max 150 steps."""

    def __init__(self, max_turns: int = 150) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-floor3-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX DUNGEON FLOOR 3\n\n"
            "Deep dungeon floor with dangerous enemies and a mage boss.\n"
            "GOAL: Find the stairs down and DESCEND to floor 4.\n"
            "You have iron gear, spells, and potions.\n"
            "REWARD: +10 for descending.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._current_floor = 3
        up_pos = self._stairs_up_pos.get(3)
        if up_pos:
            self._agent_x, self._agent_y = up_pos
        else:
            self._agent_x = _DUNGEON_SIZE // 2
            self._agent_y = _DUNGEON_SIZE // 2
        self._inventory = {
            "iron_sword": 1,
            "iron_armor": 1,
            "wood": 5,
            "coal": 5,
        }
        self._spells_learned = 3
        self._potions = ["health", "health"]
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_floor = self._current_floor
        obs, reward, terminated, truncated, info = super()._step(action)
        if self._current_floor == 4 and old_floor == 3:
            reward += 10.0
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxBossFightEnv(CraftaxFullEnv):
    """Final boss fight. Start with diamond gear on a boss floor.
    Goal: defeat the boss. Max 200 steps."""

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-bossfight-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX BOSS FIGHT\n\n"
            "You face a powerful boss (W) in the dungeon.\n"
            "GOAL: Defeat the boss.\n"
            "You have diamond sword, diamond armor, spells, and potions.\n"
            "REWARD: +10 for defeating the boss.\n"
            "Use DO facing the boss to attack. Use CAST_FIREBALL for area damage.\n"
            "Use CAST_HEAL or DRINK_POTION to restore HP.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        # Go to floor 5 (the deepest with the lich boss)
        self._current_floor = 5
        up_pos = self._stairs_up_pos.get(5)
        if up_pos:
            self._agent_x, self._agent_y = up_pos
        else:
            self._agent_x = _DUNGEON_SIZE // 2
            self._agent_y = _DUNGEON_SIZE // 2
        self._inventory = {
            "diamond_sword": 1,
            "diamond_armor": 1,
            "wood": 3,
            "coal": 3,
        }
        self._weapon_enchanted = True
        self._armor_enchanted = True
        self._spells_learned = 3
        self._potions = ["health", "health", "health"]
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        # Track initial boss count
        self._boss_count_at_start = sum(
            1 for m in self._mobs
            if m["is_boss"] and m["floor"] == 5
        )
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        # Check if boss on floor 5 is dead
        if not self._bosses_alive.get(5, True):
            reward += 10.0
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


# ===================================================================
# 5-8: Survival Tasks (Classic env)
# ===================================================================

class CraftaxSurviveHungerEnv(CraftaxClassicEnv):
    """Start with food=2, no food nearby. Must find food.
    Reward: +1 per food eaten. +10 if survive 100 steps."""

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)
        self._food_eaten: int = 0

    def env_id(self) -> str:
        return "glyphbench/craftax-survive-hunger-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX SURVIVE HUNGER\n\n"
            "You are starving! Food is at 2/9.\n"
            "GOAL: Find food before you starve. Survive 100 steps.\n"
            "- Kill a cow (c) with DO to get +5 food.\n"
            "- Eat a ripe plant (*) with EAT_PLANT for +3 food.\n"
            "- Food drains 1 every 50 steps. At 0 food: -1 HP/step.\n"
            "REWARD: +1 per food source eaten, +10 for surviving 100 steps.\n"
            "You have a wood sword to help fight cows.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        obs = super()._reset(seed)
        self._food = 2
        self._food_eaten = 0
        # Remove nearby food sources (ripe plants)
        cx, cy = self._agent_x, self._agent_y
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                x, y = cx + dx, cy + dy
                if (
                    0 <= x < self._WORLD_SIZE
                    and 0 <= y < self._WORLD_SIZE
                    and self._world[y][x] == TILE_RIPE_PLANT
                ):
                    self._world[y][x] = TILE_GRASS
        self._inventory = {"wood_sword": 1}
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_food = self._food
        obs, reward, terminated, truncated, info = super()._step(action)
        # Detect food gain
        if self._food > old_food:
            gained = self._food - old_food
            food_reward = gained / 3.0  # normalize: +1 per ~3 food gained
            reward += food_reward
            self._food_eaten += 1
        # Survival bonus at end
        if truncated and self._hp > 0:
            reward += 10.0
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxSurviveThirstEnv(CraftaxClassicEnv):
    """Start with water=2. Must find water source. Max 100 steps."""

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-survive-thirst-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX SURVIVE THIRST\n\n"
            "You are dehydrated! Water is at 2/9.\n"
            "GOAL: Find a water tile (~) and use DRINK_WATER facing it.\n"
            "Survive 100 steps without dying of thirst.\n"
            "- Water drains 1 every 40 steps. At 0 water: -1 HP/step.\n"
            "REWARD: +5 per drink, +10 for surviving 100 steps.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        obs = super()._reset(seed)
        self._water = 2
        # Remove water tiles in immediate vicinity to force exploration
        cx, cy = self._agent_x, self._agent_y
        for dx in range(-6, 7):
            for dy in range(-6, 7):
                x, y = cx + dx, cy + dy
                if (
                    0 <= x < self._WORLD_SIZE
                    and 0 <= y < self._WORLD_SIZE
                    and self._world[y][x] == TILE_WATER
                ):
                    self._world[y][x] = TILE_GRASS
        self._inventory = {}
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_water = self._water
        obs, reward, terminated, truncated, info = super()._step(action)
        if self._water > old_water:
            reward += 5.0
        if truncated and self._hp > 0:
            reward += 10.0
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxSurviveNightEnv(CraftaxClassicEnv):
    """Start at dusk. Survive the night with basic gear. Max 50 steps.
    +10 if alive at dawn."""

    def __init__(self, max_turns: int = 150) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-survive-night-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX SURVIVE THE NIGHT\n\n"
            "Night is falling! Monsters will spawn.\n"
            "GOAL: Survive until dawn.\n"
            "You have a stone sword and some stone to build shelter.\n"
            "- Hostile mobs (zombies z, skeletons k) spawn at night.\n"
            "- Place stone blocks around yourself for protection.\n"
            "- Or fight the monsters with DO.\n"
            "REWARD: +10 if alive when the sun rises.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        obs = super()._reset(seed)
        # Set time to just before nightfall
        self._day_counter = _DAY_LENGTH - 2
        self._day_night = "day"
        self._inventory = {
            "stone_sword": 1,
            "stone": 8,
        }
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._survived_night = False
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_phase = self._day_night
        obs, reward, terminated, truncated, info = super()._step(action)
        # Track if we survived through night to dawn
        if old_phase == "night" and self._day_night == "day" and self._hp > 0:
            reward += 10.0
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxSurviveWildEnv(CraftaxClassicEnv):
    """Start with nothing, day 1. Survive 200 steps managing all needs."""

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-survive-wild-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX WILDERNESS SURVIVAL\n\n"
            "You start with nothing. Survive as long as possible.\n"
            "GOAL: Survive 200 steps. Manage hunger, thirst, energy, "
            "and nighttime monsters.\n"
            "- Chop trees (DO facing tree) for wood.\n"
            "- Craft tools at a crafting table.\n"
            "- Find water (~) and drink (DRINK_WATER).\n"
            "- Kill cows or eat plants for food.\n"
            "- Build shelter before night.\n"
            "REWARD: +0.05 per step survived. +10 bonus for full 200 steps.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        obs = super()._reset(seed)
        self._inventory = {}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        # Per-step survival reward
        if not terminated:
            reward += 0.05
        # Survival bonus
        if truncated and self._hp > 0:
            reward += 10.0
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


# ===================================================================
# 9-14: Combat Tasks per Mob Type
# ===================================================================

class CraftaxFightCowEnv(CraftaxClassicEnv):
    """Start with wood sword. 1 cow nearby. Kill for food. Max 20 steps."""

    def __init__(self, max_turns: int = 20) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-fight-cow-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX FIGHT COW\n\n"
            "A cow (c) is nearby. Kill it for food.\n"
            "GOAL: Defeat the cow.\n"
            "Face the cow and use DO to attack.\n"
            "You have a wood sword (+1 damage).\n"
            "REWARD: +5 for defeating the cow.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        obs = super()._reset(seed)
        # Remove all existing mobs
        self._mobs = []
        # Place exactly 1 cow nearby
        _place_classic_mob(self, "cow", self._agent_x, self._agent_y, radius=3)
        self._inventory = {"wood_sword": 1}
        self._hp = 9
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        cow_count_before = sum(1 for m in self._mobs if m["type"] == "cow")
        obs, reward, terminated, truncated, info = super()._step(action)
        cow_count_after = sum(1 for m in self._mobs if m["type"] == "cow")
        if cow_count_after < cow_count_before:
            reward += 5.0
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxFightZombiesEnv(CraftaxClassicEnv):
    """Start with stone sword. 3 zombies in arena. Kill all. Max 40 steps."""

    def __init__(self, max_turns: int = 40) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-fight-zombies-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX FIGHT ZOMBIES\n\n"
            "3 zombies (z) are closing in!\n"
            "GOAL: Defeat all 3 zombies.\n"
            "Face a zombie and use DO to attack.\n"
            "Stone sword: +2 damage per hit. Zombie HP: 3.\n"
            "Zombies deal 1 damage when adjacent.\n"
            "REWARD: +3 per zombie killed. +5 bonus for all 3.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        obs = super()._reset(seed)
        self._mobs = []
        for _ in range(3):
            _place_classic_mob(
                self, "zombie", self._agent_x, self._agent_y, radius=4
            )
        self._inventory = {"stone_sword": 1}
        self._hp = 9
        # Disable survival drain for focused combat
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        zombie_count_before = sum(
            1 for m in self._mobs if m["type"] == "zombie"
        )
        obs, reward, terminated, truncated, info = super()._step(action)
        zombie_count_after = sum(
            1 for m in self._mobs if m["type"] == "zombie"
        )
        kills = zombie_count_before - zombie_count_after
        reward += kills * 3.0
        if zombie_count_after == 0 and zombie_count_before > 0:
            reward += 5.0
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxFightSkeletonsEnv(CraftaxClassicEnv):
    """Start with iron sword. 3 skeletons. Must close distance. Max 40 steps."""

    def __init__(self, max_turns: int = 40) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-fight-skeletons-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX FIGHT SKELETONS\n\n"
            "3 skeletons (k) are prowling nearby!\n"
            "GOAL: Defeat all 3 skeletons.\n"
            "Skeletons have 4 HP and deal 2 damage.\n"
            "You have an iron sword (+3 damage). Close distance and attack.\n"
            "REWARD: +3 per skeleton killed. +5 bonus for all 3.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        obs = super()._reset(seed)
        self._mobs = []
        for _ in range(3):
            _place_classic_mob(
                self, "skeleton", self._agent_x, self._agent_y, radius=5
            )
        self._inventory = {"iron_sword": 1}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        skel_count_before = sum(
            1 for m in self._mobs if m["type"] == "skeleton"
        )
        obs, reward, terminated, truncated, info = super()._step(action)
        skel_count_after = sum(
            1 for m in self._mobs if m["type"] == "skeleton"
        )
        kills = skel_count_before - skel_count_after
        reward += kills * 3.0
        if skel_count_after == 0 and skel_count_before > 0:
            reward += 5.0
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxFightArchersEnv(CraftaxFullEnv):
    """Start with iron sword + spells. 3 skeleton archers at range.
    Ranged combat. Max 40 steps. (Full version -- has skeleton_archer.)"""

    def __init__(self, max_turns: int = 40) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-fight-archers-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX FIGHT SKELETON ARCHERS\n\n"
            "3 skeleton archers (a) are attacking from range!\n"
            "GOAL: Defeat all 3 skeleton archers.\n"
            "Archers have 5 HP, deal 3 melee or ranged damage.\n"
            "You have an iron sword (+3 dmg) and spells.\n"
            "CAST_FIREBALL hits all mobs within 2 tiles (3 mana, 4 dmg).\n"
            "Close distance and use DO to melee attack.\n"
            "REWARD: +3 per archer killed. +5 bonus for all 3.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        # Set up an arena on the surface
        cx, cy = _SURFACE_SIZE // 2, _SURFACE_SIZE // 2
        _clear_area(self._floors[0], cx, cy, 8, _SURFACE_SIZE)
        self._current_floor = 0
        self._agent_x = cx
        self._agent_y = cy
        # Remove all mobs, then place archers
        self._mobs = []
        for _ in range(3):
            _place_full_mob(
                self, "skeleton_archer", cx, cy, floor=0, radius=5
            )
        self._inventory = {"iron_sword": 1}
        self._spells_learned = 3
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        archer_count_before = sum(
            1 for m in self._mobs if m["type"] == "skeleton_archer"
        )
        obs, reward, terminated, truncated, info = super()._step(action)
        archer_count_after = sum(
            1 for m in self._mobs if m["type"] == "skeleton_archer"
        )
        kills = archer_count_before - archer_count_after
        reward += kills * 3.0
        if archer_count_after == 0 and archer_count_before > 0:
            reward += 5.0
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxFightSpidersEnv(CraftaxFullEnv):
    """Start with iron sword. 3 spiders. Max 40 steps. (Full version.)"""

    def __init__(self, max_turns: int = 40) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-fight-spiders-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX FIGHT SPIDERS\n\n"
            "3 spiders (x) lurk nearby!\n"
            "GOAL: Defeat all 3 spiders.\n"
            "Spiders have 4 HP and deal 2 damage.\n"
            "You have an iron sword (+3 damage).\n"
            "REWARD: +3 per spider killed. +5 bonus for all 3.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        cx, cy = _SURFACE_SIZE // 2, _SURFACE_SIZE // 2
        _clear_area(self._floors[0], cx, cy, 6, _SURFACE_SIZE)
        self._current_floor = 0
        self._agent_x = cx
        self._agent_y = cy
        self._mobs = []
        for _ in range(3):
            _place_full_mob(self, "spider", cx, cy, floor=0, radius=4)
        self._inventory = {"iron_sword": 1}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        spider_count_before = sum(
            1 for m in self._mobs if m["type"] == "spider"
        )
        obs, reward, terminated, truncated, info = super()._step(action)
        spider_count_after = sum(
            1 for m in self._mobs if m["type"] == "spider"
        )
        kills = spider_count_before - spider_count_after
        reward += kills * 3.0
        if spider_count_after == 0 and spider_count_before > 0:
            reward += 5.0
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxFightBatsEnv(CraftaxFullEnv):
    """Start with stone sword. 5 bats (fast, low HP). Max 30 steps."""

    def __init__(self, max_turns: int = 30) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-fight-bats-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX FIGHT BATS\n\n"
            "5 bats (b) are swarming you!\n"
            "GOAL: Defeat all 5 bats.\n"
            "Bats have only 2 HP but move erratically.\n"
            "You have a stone sword (+2 damage).\n"
            "Face a bat and use DO to attack.\n"
            "REWARD: +2 per bat killed. +5 bonus for all 5.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        cx, cy = _SURFACE_SIZE // 2, _SURFACE_SIZE // 2
        _clear_area(self._floors[0], cx, cy, 5, _SURFACE_SIZE)
        self._current_floor = 0
        self._agent_x = cx
        self._agent_y = cy
        self._mobs = []
        for _ in range(5):
            _place_full_mob(self, "bat", cx, cy, floor=0, radius=3)
        self._inventory = {"stone_sword": 1}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        bat_count_before = sum(
            1 for m in self._mobs if m["type"] == "bat"
        )
        obs, reward, terminated, truncated, info = super()._step(action)
        bat_count_after = sum(
            1 for m in self._mobs if m["type"] == "bat"
        )
        kills = bat_count_before - bat_count_after
        reward += kills * 2.0
        if bat_count_after == 0 and bat_count_before > 0:
            reward += 5.0
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


# ===================================================================
# 15-18: Crafting Chain Tasks (Classic env)
# ===================================================================

class CraftaxCraftIronSetEnv(CraftaxClassicEnv):
    """Start near trees, stone, iron, table, furnace.
    Goal: craft iron pickaxe + iron sword. Max 150 steps."""

    def __init__(self, max_turns: int = 150) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-craft-ironset-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX CRAFT IRON SET\n\n"
            "Resources are nearby. Complete the crafting chain.\n"
            "GOAL: Craft an iron pickaxe AND an iron sword.\n"
            "CHAIN: chop wood -> place table -> craft wood pickaxe -> "
            "mine stone -> place furnace -> craft stone pickaxe -> "
            "mine iron -> craft iron tools (need table+furnace adjacent).\n"
            "REWARD: +5 for iron pickaxe, +5 for iron sword.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        obs = super()._reset(seed)
        self._mobs = []  # No mobs for crafting focus
        cx, cy = self._agent_x, self._agent_y
        # Clear area and place resources nearby
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                x, y = cx + dx, cy + dy
                if 0 <= x < self._WORLD_SIZE and 0 <= y < self._WORLD_SIZE:
                    self._world[y][x] = TILE_GRASS
        # Trees around player
        for pos in [(-3, -2), (-3, -1), (-3, 0), (-2, -3), (-1, -3)]:
            x, y = cx + pos[0], cy + pos[1]
            if 0 <= x < self._WORLD_SIZE and 0 <= y < self._WORLD_SIZE:
                self._world[y][x] = TILE_TREE
        # Stone deposit
        for pos in [(3, -2), (3, -1), (3, 0), (3, 1), (4, -1), (4, 0)]:
            x, y = cx + pos[0], cy + pos[1]
            if 0 <= x < self._WORLD_SIZE and 0 <= y < self._WORLD_SIZE:
                self._world[y][x] = TILE_STONE
        # Iron deposit
        for pos in [(0, 4), (1, 4), (-1, 4), (0, 5)]:
            x, y = cx + pos[0], cy + pos[1]
            if 0 <= x < self._WORLD_SIZE and 0 <= y < self._WORLD_SIZE:
                self._world[y][x] = TILE_IRON
        self._inventory = {}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._has_iron_pick = False
        self._has_iron_sword = False
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        # Track crafting milestones
        bonus = 0.0
        if self._inventory.get("iron_pickaxe", 0) > 0 and not self._has_iron_pick:
            self._has_iron_pick = True
            bonus += 5.0
        if self._inventory.get("iron_sword", 0) > 0 and not self._has_iron_sword:
            self._has_iron_sword = True
            bonus += 5.0
        reward += bonus
        if self._has_iron_pick and self._has_iron_sword:
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxSmeltIronEnv(CraftaxClassicEnv):
    """Start with iron ore + coal + furnace nearby.
    Goal: smelt 3 iron bars. Max 30 steps.

    Note: In this Craftax implementation, iron ore IS iron (no smelting step).
    This task instead requires mining 3 iron ore with the right pickaxe.
    """

    def __init__(self, max_turns: int = 30) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-smelt-iron-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX MINE IRON\n\n"
            "Iron deposits are nearby. You have a stone pickaxe.\n"
            "GOAL: Mine 3 iron ore.\n"
            "Face iron tiles (I) and use DO to mine.\n"
            "Requires stone pickaxe or better.\n"
            "REWARD: +3 per iron mined. +5 bonus for 3 total.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        obs = super()._reset(seed)
        self._mobs = []
        cx, cy = self._agent_x, self._agent_y
        # Clear and place iron nearby
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                x, y = cx + dx, cy + dy
                if 0 <= x < self._WORLD_SIZE and 0 <= y < self._WORLD_SIZE:
                    self._world[y][x] = TILE_GRASS
        # Iron tiles adjacent to player
        for pos in [(1, 0), (2, 0), (0, 1), (0, 2), (-1, 0)]:
            x, y = cx + pos[0], cy + pos[1]
            if 0 <= x < self._WORLD_SIZE and 0 <= y < self._WORLD_SIZE:
                self._world[y][x] = TILE_IRON
        self._inventory = {"stone_pickaxe": 1}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_iron = self._inventory.get("iron", 0)
        obs, reward, terminated, truncated, info = super()._step(action)
        new_iron = self._inventory.get("iron", 0)
        if new_iron > old_iron:
            reward += 3.0
        if new_iron >= 3:
            reward += 5.0
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxBuildShelterEnv(CraftaxClassicEnv):
    """Start with 10 stone. Place stone blocks to surround yourself.
    Goal: enclose yourself (4 adjacent placed stones). Max 50 steps."""

    def __init__(self, max_turns: int = 50) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-build-shelter-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX BUILD SHELTER\n\n"
            "You need to build a shelter before night falls!\n"
            "GOAL: Surround yourself with placed stone on all 4 sides.\n"
            "Face an empty grass tile and use PLACE_STONE.\n"
            "You need stones on LEFT, RIGHT, UP, and DOWN.\n"
            "You start with 10 stone blocks.\n"
            "REWARD: +10 for completing the shelter.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        obs = super()._reset(seed)
        self._mobs = []
        cx, cy = self._agent_x, self._agent_y
        # Clear the area
        _clear_area(self._world, cx, cy, 5, self._WORLD_SIZE)
        self._inventory = {"stone": 10}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        # Check if shelter is complete (placed stone on all 4 sides)
        if self._has_shelter():
            reward += 10.0
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxPlantFarmEnv(CraftaxClassicEnv):
    """Start with saplings. Plant, grow, harvest.
    Goal: eat 3 ripe plants. Max 100 steps."""

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)
        self._plants_harvested: int = 0

    def env_id(self) -> str:
        return "glyphbench/craftax-plant-farm-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX PLANT FARM\n\n"
            "You have saplings to plant.\n"
            "GOAL: Harvest (eat) 3 ripe plants.\n"
            "1. Face grass and PLACE_PLANT to plant a sapling (;).\n"
            "2. Wait 20 steps for it to ripen into (*) ripe plant.\n"
            "3. Face ripe plant and EAT_PLANT to harvest.\n"
            "You start with 5 saplings.\n"
            "REWARD: +3 per plant eaten. +5 bonus for 3 total.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        obs = super()._reset(seed)
        self._mobs = []
        self._plants_harvested = 0
        cx, cy = self._agent_x, self._agent_y
        _clear_area(self._world, cx, cy, 6, self._WORLD_SIZE)
        self._inventory = {"sapling": 5}
        self._hp = 9
        self._food = 5  # Not full, so eating is useful
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_food = self._food
        obs, reward, terminated, truncated, info = super()._step(action)
        # Detect plant eating (food increases by 3 per plant)
        if self._food > old_food and self.action_spec.names[action] == "EAT_PLANT":
            self._plants_harvested += 1
            reward += 3.0
        if self._plants_harvested >= 3:
            reward += 5.0
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


# ===================================================================
# 19-21: Exploration Tasks
# ===================================================================

class CraftaxFindWaterEnv(CraftaxClassicEnv):
    """Start in a desert-like area. Navigate to find water. Max 100 steps."""

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-find-water-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX FIND WATER\n\n"
            "You are in a dry area. Water is scarce.\n"
            "GOAL: Find a water tile (~) and drink (DRINK_WATER).\n"
            "Move in all directions to explore. Water is somewhere\n"
            "in the world but not near your starting position.\n"
            "REWARD: +10 for drinking water.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        obs = super()._reset(seed)
        self._mobs = []
        cx, cy = self._agent_x, self._agent_y
        # Convert nearby water and grass to sand (desert)
        for dx in range(-12, 13):
            for dy in range(-12, 13):
                x, y = cx + dx, cy + dy
                if (
                    0 <= x < self._WORLD_SIZE
                    and 0 <= y < self._WORLD_SIZE
                ):
                    tile = self._world[y][x]
                    if tile == TILE_WATER:
                        self._world[y][x] = TILE_SAND
                    elif tile == TILE_GRASS:
                        self._world[y][x] = TILE_SAND
        # Ensure walkable start
        self._world[cy][cx] = TILE_SAND
        self._inventory = {}
        self._hp = 9
        self._water = 5  # Moderate thirst
        self._food = _MAX_FOOD
        self._energy = _MAX_ENERGY
        self._found_water = False
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_water = self._water
        obs, reward, terminated, truncated, info = super()._step(action)
        if self._water > old_water and not self._found_water:
            self._found_water = True
            reward += 10.0
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxFindDiamondEnv(CraftaxClassicEnv):
    """Start with iron pickaxe. Find and mine a diamond. Max 150 steps."""

    def __init__(self, max_turns: int = 150) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-find-diamond-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX FIND DIAMOND\n\n"
            "Diamonds (D) are rare and scattered across the world.\n"
            "GOAL: Find a diamond tile and mine it with DO.\n"
            "You have an iron pickaxe (required to mine diamond).\n"
            "Explore widely -- diamonds may be far from your start.\n"
            "REWARD: +10 for mining a diamond.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        obs = super()._reset(seed)
        self._mobs = []
        self._inventory = {"iron_pickaxe": 1}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_diamond = self._inventory.get("diamond", 0)
        obs, reward, terminated, truncated, info = super()._step(action)
        new_diamond = self._inventory.get("diamond", 0)
        if new_diamond > old_diamond:
            reward += 10.0
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


class CraftaxReachDungeonEnv(CraftaxFullEnv):
    """Start on surface. Find the dungeon entrance (stairs down).
    Max 100 steps."""

    def __init__(self, max_turns: int = 100) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-reach-dungeon-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX REACH THE DUNGEON\n\n"
            "A dungeon entrance is somewhere on the surface.\n"
            "GOAL: Find the stairs down and DESCEND.\n"
            "Look for the stairs down tile. It is placed near the\n"
            "center of the map but you must navigate to it.\n"
            "REWARD: +10 for entering the dungeon.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._current_floor = 0
        # Start a few tiles away from dungeon entrance
        entrance = self._stairs_down_pos.get(0)
        if entrance:
            # Offset start position
            self._agent_x = max(5, entrance[0] - 8)
            self._agent_y = max(5, entrance[1] - 8)
        self._inventory = {"stone_sword": 1}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        old_floor = self._current_floor
        obs, reward, terminated, truncated, info = super()._step(action)
        if self._current_floor == 1 and old_floor == 0:
            reward += 10.0
            terminated = True
            info["subtask_success"] = True
        return obs, reward, terminated, truncated, info


# ===================================================================
# 22-23: Economy/Achievement Tasks
# ===================================================================

class CraftaxFirstDayEnv(CraftaxClassicEnv):
    """Complete as many achievements as possible in 1 day cycle (200 steps).
    Start from scratch. Reward: +1 per achievement unlocked."""

    def __init__(self, max_turns: int = 200) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-firstday-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX FIRST DAY CHALLENGE\n\n"
            "You have one day (200 steps) to accomplish as much as possible.\n"
            "GOAL: Unlock as many achievements as you can.\n"
            "Start from scratch. Gather resources, craft tools, fight mobs.\n"
            "Each achievement gives +1 reward:\n"
            "collect_wood, place_table, make_wood_pickaxe, collect_stone,\n"
            "place_furnace, make_stone_pickaxe, collect_iron, collect_coal,\n"
            "place_stone, collect_drink, collect_sapling, place_plant,\n"
            "eat_plant, defeat_zombie, defeat_skeleton, wake_up,\n"
            "collect_diamond, make_iron_pickaxe, make_iron_sword,\n"
            "make_wood_sword, make_stone_sword, eat_cow.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        obs = super()._reset(seed)
        self._inventory = {}
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        # Achievement reward is already handled by parent (+1 per unlock)
        info["total_achievements"] = len(self._achievements_unlocked)
        # End at nightfall
        if self._day_night == "night" and not terminated:
            truncated = True
            info["subtask_reason"] = "night_fell"
        return obs, reward, terminated, truncated, info


class CraftaxSpeedrunEnv(CraftaxFullEnv):
    """Reach the boss as fast as possible. Start with endgame gear.
    Navigate through dungeon floors. Max 300 steps."""

    def __init__(self, max_turns: int = 300) -> None:
        super().__init__(max_turns=max_turns)

    def env_id(self) -> str:
        return "glyphbench/craftax-speedrun-v0"

    def system_prompt(self) -> str:
        return (
            "CRAFTAX SPEEDRUN\n\n"
            "Race to the deepest dungeon floor!\n"
            "GOAL: Descend as many floors as possible.\n"
            "You start on the surface with endgame gear.\n"
            "Find stairs down, DESCEND, repeat.\n"
            "Each floor reached gives +3 reward.\n"
            "Reaching floor 5 gives +10 bonus.\n"
            "You have diamond gear, spells, and torches.\n\n"
            + self.action_spec.render_for_prompt()
        )

    def _reset(self, seed: int) -> GridObservation:
        super()._reset(seed)
        self._current_floor = 0
        # Start near dungeon entrance
        entrance = self._stairs_down_pos.get(0)
        if entrance:
            self._agent_x = entrance[0]
            self._agent_y = entrance[1]
        self._inventory = {
            "diamond_sword": 1,
            "diamond_armor": 1,
            "wood": 10,
            "coal": 10,
        }
        self._weapon_enchanted = True
        self._armor_enchanted = True
        self._spells_learned = 3
        self._potions = ["health", "health", "speed"]
        self._hp = 9
        self._food = _MAX_FOOD
        self._water = _MAX_WATER
        self._energy = _MAX_ENERGY
        self._mana = _MAX_MANA
        self._deepest_floor = 0
        return self._render_current_observation()

    def _step(
        self, action: int,
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super()._step(action)
        # Reward for reaching new floors
        if self._current_floor > self._deepest_floor:
            floors_gained = self._current_floor - self._deepest_floor
            self._deepest_floor = self._current_floor
            reward += floors_gained * 3.0
            if self._current_floor >= 5:
                reward += 10.0
                terminated = True
                info["subtask_success"] = True
        info["deepest_floor"] = self._deepest_floor
        return obs, reward, terminated, truncated, info


# Registration is handled in glyphbench.envs.craftax.__init__.
