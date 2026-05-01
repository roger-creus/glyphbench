"""Craftax focused sub-task environments.

Each sub-task starts the player in a curated initial state and has a clear
success condition, testing a specific skill (navigation, crafting, combat,
dungeon exploration).

All 10 environments subclass CraftaxClassicEnv to reuse its movement,
crafting, combat, and rendering machinery.  Only ``_reset`` (curated world
setup) and ``_step`` (sub-task termination / reward shaping) are overridden.
"""

from __future__ import annotations

from typing import Any

from glyphbench.core.glyph_primitives import build_legend, grid_to_string
from glyphbench.core.observation import GridObservation
from glyphbench.envs.craftax.base import (
    CRAFTAX_ACTION_SPEC,
    TILE_DUNGEON_FLOOR,
    TILE_DUNGEON_WALL,
    TILE_FURNACE,
    TILE_GRASS,
    TILE_SKELETON,
    TILE_STAIRS_DOWN,
    TILE_STONE,
    TILE_TABLE,
    TILE_TREE,
    TILE_ZOMBIE,
    VIEW_HEIGHT,
    VIEW_WIDTH,
    _CraftaxTutorialMixin,
)
from glyphbench.envs.craftaxfull.classic import (
    WALKABLE_TILES,
    CraftaxClassicEnv,
    Mob,
    _MOB_STATS,
    _MOB_TILES,
)

# ---------------------------------------------------------------------------
# Helpers shared across sub-tasks
# ---------------------------------------------------------------------------

_SUBTASK_WORLD_SIZE = 16  # small curated arenas


def _blank_world(size: int = _SUBTASK_WORLD_SIZE, fill: str = TILE_GRASS) -> list[list[str]]:
    """Return a square grid filled with *fill*."""
    return [[fill for _ in range(size)] for _ in range(size)]


def _place_ring(world: list[list[str]], cx: int, cy: int, radius: int, tile: str) -> None:
    """Place *tile* in a ring around (cx, cy) at Manhattan distance == radius."""
    size = len(world)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if abs(dx) + abs(dy) == radius:
                x, y = cx + dx, cy + dy
                if 0 <= x < size and 0 <= y < size:
                    world[y][x] = tile


def _spawn_mob(mobs: list[Mob], mob_type: str, x: int, y: int) -> None:
    """Append a fresh mob of *mob_type* at (x, y)."""
    stats = _MOB_STATS[mob_type]
    mobs.append(Mob(type=mob_type, x=x, y=y, hp=stats["hp"], max_hp=stats["hp"]))


# ===================================================================
# Base mixin that patches CraftaxClassicEnv for small-arena sub-tasks
# ===================================================================

class _SubtaskMixin:
    """Shared overrides for all Craftax sub-task envs.

    Must be listed *before* CraftaxClassicEnv in the MRO so that its
    ``_reset`` / ``_step`` run first.

    Subclasses implement:
    - ``_setup_world(seed)`` -- populate ``self._world``, ``self._inventory``,
      ``self._mobs``, agent position, etc.
    - ``_subtask_check(reward, info)`` -- return ``(extra_reward, terminated)``
      after the parent ``_step`` has run.
    """

    # Sub-tasks use a small arena
    _WORLD_SIZE: int = _SUBTASK_WORLD_SIZE

    # Disable survival drains (food/water/energy never decrease) so the
    # sub-task tests only the intended skill.
    _disable_survival: bool = True
    # Disable day/night mob spawns
    _disable_day_night: bool = True

    # Subclasses set these
    _subtask_max_turns: int = 50

    def __init__(self, max_turns: int | None = None, **kwargs: Any) -> None:
        # External override wins; else use the sub-task's tight budget
        effective = max_turns if max_turns is not None else self._subtask_max_turns
        super().__init__(max_turns=effective, **kwargs)  # type: ignore[call-arg]

    # -- reset -----------------------------------------------------------

    def _reset(self, seed: int) -> GridObservation:  # type: ignore[override]
        # Minimal init (parent fields)
        self._world = _blank_world(self._WORLD_SIZE)  # type: ignore[attr-defined]
        self._agent_x = self._WORLD_SIZE // 2  # type: ignore[attr-defined]
        self._agent_y = self._WORLD_SIZE // 2  # type: ignore[attr-defined]
        self._facing = (1, 0)  # type: ignore[attr-defined]
        self._inventory = {}  # type: ignore[attr-defined]
        self._achievements_unlocked = set()  # type: ignore[attr-defined]
        self._message = ""  # type: ignore[attr-defined]
        self._hp = 9  # type: ignore[attr-defined]
        self._max_hp = 9  # type: ignore[attr-defined]
        self._food = 9  # type: ignore[attr-defined]
        self._water = 9  # type: ignore[attr-defined]
        self._energy = 9  # type: ignore[attr-defined]
        self._day_counter = 0  # type: ignore[attr-defined]
        self._day_night = "day"  # type: ignore[attr-defined]
        self._mobs = []  # type: ignore[attr-defined]
        self._plants = {}  # type: ignore[attr-defined]

        # Subclass populates the curated world
        self._setup_world(seed)  # type: ignore[attr-defined]
        return self._render_current_observation()  # type: ignore[attr-defined]

    # -- step (wraps parent, then checks sub-task goal) ------------------

    def _step(  # type: ignore[override]
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]:
        # Run the full Classic step machinery
        obs, reward, terminated, truncated, info = super()._step(action)  # type: ignore[misc]

        # Always invoke the subtask check, even when the parent already
        # terminated the episode (e.g. the agent died). Otherwise per-
        # subtask death-penalty branches are unreachable. The check is
        # responsible for keeping `terminated` True when the parent
        # already set it.
        extra_reward, done = self._subtask_check(reward, info)  # type: ignore[attr-defined]
        reward += extra_reward
        terminated = bool(terminated) or bool(done)

        return obs, reward, terminated, truncated, info

    # -- disable survival / day-night if flags set -----------------------

    def _apply_survival_drain(self) -> None:  # type: ignore[override]
        if getattr(self, "_disable_survival", False):
            return
        super()._apply_survival_drain()  # type: ignore[misc]

    def _advance_day_counter(self, steps: int = 1) -> None:  # type: ignore[override]
        if getattr(self, "_disable_day_night", False):
            # Still tick the counter (some rendering uses it) but skip
            # mob spawn / despawn logic.
            self._day_counter += steps  # type: ignore[attr-defined]
            return
        super()._advance_day_counter(steps)  # type: ignore[misc]

    # -- subclass hooks (abstract) ---------------------------------------

    def _setup_world(self, seed: int) -> None:
        raise NotImplementedError

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        """Return (extra_reward, terminated)."""
        raise NotImplementedError


# ======================================================================
# 1. CraftaxChopTreesEnv -- collect 5 wood
# ======================================================================

class CraftaxChopTreesEnv(_SubtaskMixin, CraftaxClassicEnv):
    """Start in a meadow with trees.  Goal: collect 5 wood."""

    _subtask_max_turns = 70

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-choptrees-v0"

    def _task_description(self) -> str:
        return (
            "Collect 5 wood by chopping trees. Face a tree and use DO. "
            "Reward: +1 per wood collected. Episode ends when you have 5 wood "
            "or after 70 steps."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        # Grass arena with trees around the edges and scattered inside
        for x in range(size):
            for y in range(size):
                # Border trees
                if x <= 1 or x >= size - 2 or y <= 1 or y >= size - 2:
                    self._world[y][x] = TILE_TREE
                # Scattered interior trees
                elif self.rng.random() < 0.20:
                    self._world[y][x] = TILE_TREE
        # Clear agent spawn area
        cx, cy = size // 2, size // 2
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                self._world[cy + dy][cx + dx] = TILE_GRASS
        self._agent_x = cx
        self._agent_y = cy

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        wood = self._inventory.get("wood", 0)
        # Reward is handled by the parent achievement system (+1 for first
        # wood via collect_wood), but we add shaped reward for every piece.
        # The parent already gave achievement reward; we give +1 per wood
        # collected this step beyond that.
        extra = 0.0
        if wood >= 5:
            self._message = "Goal complete! Collected 5 wood."
            return extra, True
        return extra, False

    # Override _handle_do to give +1 per wood (shaped reward)
    def _handle_do(self) -> float:
        old_wood = self._inventory.get("wood", 0)
        reward = super()._handle_do()
        new_wood = self._inventory.get("wood", 0)
        reward += (new_wood - old_wood)  # +1 per wood collected
        return reward


# ======================================================================
# 2. CraftaxMineStoneEnv -- mine 5 stone with wood pickaxe
# ======================================================================

class CraftaxMineStoneEnv(_SubtaskMixin, CraftaxClassicEnv):
    """Start with a wood pickaxe near stone deposits.  Goal: mine 5 stone."""

    _subtask_max_turns = 70

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-minestone-v0"

    def _task_description(self) -> str:
        return (
            "Mine 5 stone. You start with a wood pickaxe. Face stone (S) and use "
            "DO to mine. Reward: +1 per stone mined. Episode ends when you have "
            "5 stone or after 70 steps."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        cx, cy = size // 2, size // 2
        # Scatter stone tiles densely
        for x in range(2, size - 2):
            for y in range(2, size - 2):
                if self.rng.random() < 0.35:
                    self._world[y][x] = TILE_STONE
        # Clear agent spawn area
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                self._world[cy + dy][cx + dx] = TILE_GRASS
        self._agent_x = cx
        self._agent_y = cy
        self._inventory["wood_pickaxe"] = 1

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        stone = self._inventory.get("stone", 0)
        if stone >= 5:
            self._message = "Goal complete! Mined 5 stone."
            return 0.0, True
        return 0.0, False

    def _handle_do(self) -> float:
        old_stone = self._inventory.get("stone", 0)
        reward = super()._handle_do()
        new_stone = self._inventory.get("stone", 0)
        reward += (new_stone - old_stone)
        return reward


# ======================================================================
# 3. CraftaxGatherResourcesEnv -- get 3 wood + 3 stone (multi-step)
# ======================================================================

class CraftaxGatherResourcesEnv(_SubtaskMixin, CraftaxClassicEnv):
    """Start empty-handed.  Collect wood, craft pickaxe, mine stone.
    Goal: have 3 wood + 3 stone in inventory.  Multi-step planning."""

    _subtask_max_turns = 110

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-gatherresources-v0"

    def _task_description(self) -> str:
        return (
            "Gather wood (5+), stone (5+), and one each of coal, iron, and "
            "diamond. Reward: +1 per first-time resource collected."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        cx, cy = size // 2, size // 2
        # Trees in top half, stone in bottom half
        for x in range(2, size - 2):
            for y in range(2, cy - 1):
                if self.rng.random() < 0.30:
                    self._world[y][x] = TILE_TREE
        for x in range(2, size - 2):
            for y in range(cy + 2, size - 2):
                if self.rng.random() < 0.30:
                    self._world[y][x] = TILE_STONE
        # Clear spawn
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < size and 0 <= ny < size:
                    self._world[ny][nx] = TILE_GRASS
        self._agent_x = cx
        self._agent_y = cy

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        wood = self._inventory.get("wood", 0)
        stone = self._inventory.get("stone", 0)
        if wood >= 3 and stone >= 3:
            self._message = "Goal complete! Gathered 3 wood and 3 stone."
            return 5.0, True
        return 0.0, False

    def _handle_do(self) -> float:
        old_wood = self._inventory.get("wood", 0)
        old_stone = self._inventory.get("stone", 0)
        reward = super()._handle_do()
        new_wood = self._inventory.get("wood", 0)
        new_stone = self._inventory.get("stone", 0)
        reward += (new_wood - old_wood) + (new_stone - old_stone)
        return reward


# ======================================================================
# 4. CraftaxCraftPickaxeEnv -- craft a wood pickaxe
# ======================================================================

class CraftaxCraftPickaxeEnv(_SubtaskMixin, CraftaxClassicEnv):
    """Start with 1 wood, adjacent to a crafting table.
    Goal: craft a wood pickaxe."""

    _subtask_max_turns = 30

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-craftpickaxe-v0"

    def _task_description(self) -> str:
        return (
            "Craft a wood pickaxe. You start with 1 wood and a crafting table "
            "(t) adjacent to you. Stand next to the table and use "
            "MAKE_WOOD_PICKAXE. Reward: +10 on craft, episode ends. Time limit: "
            "30 steps."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        cx, cy = size // 2, size // 2
        self._agent_x = cx
        self._agent_y = cy
        # Place table adjacent (to the right)
        self._world[cy][cx + 1] = TILE_TABLE
        self._inventory["wood"] = 1

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        if self._inventory.get("wood_pickaxe", 0) >= 1:
            self._message = "Goal complete! Crafted wood pickaxe."
            return 10.0, True
        return 0.0, False


# ======================================================================
# 5. CraftaxCraftSwordEnv -- craft a stone sword
# ======================================================================

class CraftaxCraftSwordEnv(_SubtaskMixin, CraftaxClassicEnv):
    """Start with 2 wood + 1 stone, near table + furnace.
    Goal: craft a stone sword."""

    _subtask_max_turns = 30

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-craftsword-v0"

    def _task_description(self) -> str:
        return (
            "Craft a stone sword. You start with 2 wood and 1 stone, a crafting "
            "table (t) on your right, and a furnace (f) on your left. Stand "
            "adjacent to the table and use MAKE_STONE_SWORD. Reward: +10 on "
            "craft, episode ends. Time limit: 30 steps."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        cx, cy = size // 2, size // 2
        self._agent_x = cx
        self._agent_y = cy
        # Table to the right, furnace to the left
        self._world[cy][cx + 1] = TILE_TABLE
        self._world[cy][cx - 1] = TILE_FURNACE
        self._inventory["wood"] = 2
        self._inventory["stone"] = 1

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        if self._inventory.get("stone_sword", 0) >= 1:
            self._message = "Goal complete! Crafted stone sword."
            return 10.0, True
        return 0.0, False


# ======================================================================
# 6. CraftaxCraftChainEnv -- chain of 3 crafts
# ======================================================================

class CraftaxCraftChainEnv(_SubtaskMixin, CraftaxClassicEnv):
    """Start empty near trees + crafting table + stone.
    Goal: craft wood pickaxe, mine stone, craft stone sword.
    Multi-step chain of 3 crafts."""

    _subtask_max_turns = 120

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-craftchain-v0"

    def _task_description(self) -> str:
        return (
            "Craft the full tech chain: wood pickaxe, stone pickaxe, iron "
            "pickaxe, wood sword, stone sword, iron sword. Reward: +1 per "
            "first-time craft."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        cx, cy = size // 2, size // 2
        # Trees in the north
        for x in range(3, size - 3):
            for y in range(2, cy - 2):
                if self.rng.random() < 0.35:
                    self._world[y][x] = TILE_TREE
        # Stone in the south
        for x in range(3, size - 3):
            for y in range(cy + 3, size - 2):
                if self.rng.random() < 0.35:
                    self._world[y][x] = TILE_STONE
        # Pre-place table and furnace near center
        self._world[cy][cx + 2] = TILE_TABLE
        self._world[cy][cx - 2] = TILE_FURNACE
        # Clear spawn
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = cx + dx, cy + dy
                self._world[ny][nx] = TILE_GRASS
        self._agent_x = cx
        self._agent_y = cy
        # Track milestone rewards
        self._chain_milestones: set[str] = set()

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        extra = 0.0
        if (
            "pickaxe_crafted" not in self._chain_milestones
            and self._inventory.get("wood_pickaxe", 0) >= 1
        ):
            self._chain_milestones.add("pickaxe_crafted")
            extra += 3.0
        if (
            "stone_mined" not in self._chain_milestones
            and self._inventory.get("stone", 0) >= 1
        ):
            self._chain_milestones.add("stone_mined")
            extra += 3.0
        if self._inventory.get("stone_sword", 0) >= 1:
            if "sword_crafted" not in self._chain_milestones:
                self._chain_milestones.add("sword_crafted")
                extra += 3.0
            self._message = "Goal complete! Crafted stone sword via chain."
            return extra + 5.0, True
        return extra, False


# ======================================================================
# 7. CraftaxFightZombieEnv -- kill one zombie
# ======================================================================

class CraftaxFightZombieEnv(_SubtaskMixin, CraftaxClassicEnv):
    """Start with a stone sword, one zombie nearby.
    Goal: kill it without dying."""

    _subtask_max_turns = 30

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-fightzombie-v0"

    def _task_description(self) -> str:
        return (
            "Fight one zombie spawned next to you on the overworld. Kill it with "
            "DO while facing it. Reward: +1 on kill, -10 on death. Episode ends "
            "on either."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        cx, cy = size // 2, size // 2
        self._agent_x = cx
        self._agent_y = cy
        self._inventory["stone_sword"] = 1
        # Spawn a zombie 3 tiles to the right
        _spawn_mob(self._mobs, "zombie", cx + 3, cy)

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        # Check if zombie is dead (no hostile mobs left)
        hostiles = [m for m in self._mobs if m["type"] != "cow"]
        if not hostiles:
            self._message = "Goal complete! Zombie defeated."
            return 10.0, True
        # Death handled by parent (terminated = True when hp <= 0)
        if self._hp <= 0:
            return -10.0, True
        return 0.0, False


# ======================================================================
# 8. CraftaxSurviveHordeEnv -- survive and kill 5 mobs
# ======================================================================

class CraftaxSurviveHordeEnv(_SubtaskMixin, CraftaxClassicEnv):
    """Start with iron sword.  3 zombies + 2 skeletons approaching.
    Survive and kill all."""

    _subtask_max_turns = 50

    tutorial_sections = (
        "legend:player", "legend:terrain", "legend:mobs:overworld",
        "survival:hp_food_drink", "survival:energy_sleep", "survival:day_night",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-survivehorde-v0"

    def _task_description(self) -> str:
        return (
            "Survive a wave of hostile mobs. Use DO to fight, place stone for "
            "shelter, sleep when safe to recover energy. Reward: +1 per turn "
            "survived; -10 on death."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        cx, cy = size // 2, size // 2
        self._agent_x = cx
        self._agent_y = cy
        self._inventory["iron_sword"] = 1
        self._hp = 9
        # Spawn 3 zombies and 2 skeletons in a ring around the player
        positions = [
            (cx + 3, cy),
            (cx - 3, cy),
            (cx, cy + 3),
            (cx + 2, cy + 2),
            (cx - 2, cy - 2),
        ]
        for i, (mx, my) in enumerate(positions):
            mob_type = "zombie" if i < 3 else "skeleton"
            _spawn_mob(self._mobs, mob_type, mx, my)
        self._horde_kills: int = 0

    def _handle_do(self) -> float:
        old_count = len(self._mobs)
        reward = super()._handle_do()
        new_count = len(self._mobs)
        killed = old_count - new_count
        if killed > 0:
            self._horde_kills += killed
            reward += killed * 5.0  # +5 per mob killed
        return reward

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        hostiles = [m for m in self._mobs if m["type"] != "cow"]
        if not hostiles:
            self._message = "Goal complete! Horde defeated."
            return 10.0, True
        if self._hp <= 0:
            return -10.0, True
        return 0.0, False


# ======================================================================
# 9. CraftaxDungeonExploreEnv -- find stairs down in a dungeon
# ======================================================================

# Dungeon sub-tasks use their own walkable set
_DUNGEON_WALKABLE = frozenset({
    TILE_DUNGEON_FLOOR, TILE_STAIRS_DOWN, TILE_TABLE, TILE_FURNACE,
})


class CraftaxDungeonExploreEnv(_SubtaskMixin, CraftaxClassicEnv):
    """Start at dungeon entrance.  Find stairs down.  No enemies."""

    _subtask_max_turns = 100
    _WORLD_SIZE = 24  # slightly larger for dungeon layout

    tutorial_sections = (
        "legend:player", "legend:terrain",
        "legend:mobs:overworld", "legend:mobs:dungeon",
        "legend:items",
        "survival:hp_food_drink", "survival:energy_sleep",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0", "floors:1", "floors:navigation",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-dungeonexplore-v0"

    def _task_description(self) -> str:
        return (
            "Explore the dungeon (floor 1). Find the stairs down (⇣) by "
            "navigating rooms and corridors. Reward shaped on dungeon entry "
            "and exploration progress."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        # Fill with dungeon wall
        self._world = [[TILE_DUNGEON_WALL for _ in range(size)] for _ in range(size)]

        # Generate rooms
        rooms: list[tuple[int, int, int, int]] = []
        for _ in range(int(self.rng.integers(4, 7))):
            rw = int(self.rng.integers(3, 6))
            rh = int(self.rng.integers(3, 6))
            rx = int(self.rng.integers(1, size - rw - 1))
            ry = int(self.rng.integers(1, size - rh - 1))
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
                    self._world[ry + dy][rx + dx] = TILE_DUNGEON_FLOOR

        # Ensure at least 2 rooms
        if len(rooms) < 2:
            rooms = [(2, 2, 5, 5), (size - 8, size - 8, 5, 5)]
            for rx, ry, rw, rh in rooms:
                for dy in range(rh):
                    for dx in range(rw):
                        if 0 <= ry + dy < size and 0 <= rx + dx < size:
                            self._world[ry + dy][rx + dx] = TILE_DUNGEON_FLOOR

        # Connect rooms with corridors
        for i in range(len(rooms) - 1):
            r1 = rooms[i]
            r2 = rooms[i + 1]
            cx1 = r1[0] + r1[2] // 2
            cy1 = r1[1] + r1[3] // 2
            cx2 = r2[0] + r2[2] // 2
            cy2 = r2[1] + r2[3] // 2
            x = cx1
            while x != cx2:
                if 0 <= x < size and 0 <= cy1 < size:
                    self._world[cy1][x] = TILE_DUNGEON_FLOOR
                x += 1 if cx2 > cx1 else -1
            if 0 <= cx2 < size and 0 <= cy1 < size:
                self._world[cy1][cx2] = TILE_DUNGEON_FLOOR
            y = cy1
            while y != cy2:
                if 0 <= cx2 < size and 0 <= y < size:
                    self._world[y][cx2] = TILE_DUNGEON_FLOOR
                y += 1 if cy2 > cy1 else -1
            if 0 <= cx2 < size and 0 <= cy2 < size:
                self._world[cy2][cx2] = TILE_DUNGEON_FLOOR

        # Agent starts in first room
        r0 = rooms[0]
        self._agent_x = r0[0] + r0[2] // 2
        self._agent_y = r0[1] + r0[3] // 2

        # Stairs down in last room
        rl = rooms[-1]
        self._stairs_goal = (
            rl[0] + rl[2] // 2,
            rl[1] + rl[3] // 2,
        )
        sx, sy = self._stairs_goal
        self._world[sy][sx] = TILE_STAIRS_DOWN

    # Override movement to use dungeon walkable tiles
    def _handle_move(self, name: str) -> float:
        direction_map = {
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
            and self._world[ny][nx] in _DUNGEON_WALKABLE
            and not self._mob_at(nx, ny)
        ):
            self._agent_x = nx
            self._agent_y = ny
        return 0.0

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        sx, sy = self._stairs_goal
        if self._agent_x == sx and self._agent_y == sy:
            self._message = "Goal complete! Found the stairs down."
            return 10.0, True
        return 0.0, False

    # Custom rendering for dungeon tiles
    def _render_current_observation(self) -> GridObservation:
        from glyphbench.envs.craftaxfull.classic import _DIR_CHARS, _DIR_NAMES

        half_w = VIEW_WIDTH // 2
        half_h = VIEW_HEIGHT // 2
        grid: list[list[str]] = []
        symbols_seen: set[str] = set()
        facing_ch = _DIR_CHARS.get(self._facing, "@")
        facing_name = _DIR_NAMES.get(self._facing, "right")

        for wy in range(VIEW_HEIGHT):
            row: list[str] = []
            for wx in range(VIEW_WIDTH):
                world_x = self._agent_x - half_w + wx
                world_y = self._agent_y - half_h + wy
                if world_x == self._agent_x and world_y == self._agent_y:
                    row.append(facing_ch)
                    symbols_seen.add(facing_ch)
                elif (
                    0 <= world_x < self._WORLD_SIZE
                    and 0 <= world_y < self._WORLD_SIZE
                ):
                    tile = self._world[world_y][world_x]
                    row.append(tile)
                    symbols_seen.add(tile)
                else:
                    row.append(TILE_DUNGEON_WALL)
                    symbols_seen.add(TILE_DUNGEON_WALL)
            grid.append(row)

        hud = f"Step: {self._turn} / {self.max_turns}  HP: {self._hp}/{self._max_hp}"

        tile_meanings: dict[str, str] = {
            TILE_DUNGEON_WALL: "dungeon wall (impassable)",
            TILE_DUNGEON_FLOOR: "dungeon floor",
            TILE_STAIRS_DOWN: "stairs down (GOAL)",
        }
        legend_entries: dict[str, str] = {}
        agent_legend = f"you (facing {facing_name})"
        for sym in symbols_seen:
            if sym == facing_ch:
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


# ======================================================================
# 10. CraftaxDungeonClearEnv -- kill all enemies on a dungeon floor
# ======================================================================

class CraftaxDungeonClearEnv(_SubtaskMixin, CraftaxClassicEnv):
    """Start in dungeon with a sword.  Kill all enemies on the floor."""

    _subtask_max_turns = 120
    _WORLD_SIZE = 24

    tutorial_sections = (
        "legend:player", "legend:terrain",
        "legend:mobs:overworld", "legend:mobs:dungeon",
        "legend:items",
        "survival:hp_food_drink", "survival:energy_sleep",
        "combat:melee",
        "crafting:wood", "crafting:stone", "crafting:iron",
        "crafting:placement",
        "items:resources",
        "floors:0", "floors:1", "floors:navigation",
    )

    def env_id(self) -> str:
        return "glyphbench/craftax-dungeonclear-v0"

    def _task_description(self) -> str:
        return (
            "Clear all hostile mobs on dungeon floor 1. Reward: +1 per mob "
            "killed; -10 on death. Episode ends when all mobs are dead or you "
            "die."
        )

    def _setup_world(self, seed: int) -> None:
        size = self._WORLD_SIZE
        # Fill with dungeon wall
        self._world = [[TILE_DUNGEON_WALL for _ in range(size)] for _ in range(size)]

        # Generate rooms
        rooms: list[tuple[int, int, int, int]] = []
        for _ in range(int(self.rng.integers(4, 7))):
            rw = int(self.rng.integers(4, 7))
            rh = int(self.rng.integers(4, 7))
            rx = int(self.rng.integers(1, size - rw - 1))
            ry = int(self.rng.integers(1, size - rh - 1))
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
                    self._world[ry + dy][rx + dx] = TILE_DUNGEON_FLOOR

        # Ensure at least 2 rooms
        if len(rooms) < 2:
            rooms = [(2, 2, 6, 6), (size - 9, size - 9, 6, 6)]
            for rx, ry, rw, rh in rooms:
                for dy in range(rh):
                    for dx in range(rw):
                        if 0 <= ry + dy < size and 0 <= rx + dx < size:
                            self._world[ry + dy][rx + dx] = TILE_DUNGEON_FLOOR

        # Connect rooms with corridors
        for i in range(len(rooms) - 1):
            r1 = rooms[i]
            r2 = rooms[i + 1]
            cx1 = r1[0] + r1[2] // 2
            cy1 = r1[1] + r1[3] // 2
            cx2 = r2[0] + r2[2] // 2
            cy2 = r2[1] + r2[3] // 2
            x = cx1
            while x != cx2:
                if 0 <= x < size and 0 <= cy1 < size:
                    self._world[cy1][x] = TILE_DUNGEON_FLOOR
                x += 1 if cx2 > cx1 else -1
            if 0 <= cx2 < size and 0 <= cy1 < size:
                self._world[cy1][cx2] = TILE_DUNGEON_FLOOR
            y = cy1
            while y != cy2:
                if 0 <= cx2 < size and 0 <= y < size:
                    self._world[y][cx2] = TILE_DUNGEON_FLOOR
                y += 1 if cy2 > cy1 else -1
            if 0 <= cx2 < size and 0 <= cy2 < size:
                self._world[cy2][cx2] = TILE_DUNGEON_FLOOR

        # Agent starts in first room
        r0 = rooms[0]
        self._agent_x = r0[0] + r0[2] // 2
        self._agent_y = r0[1] + r0[3] // 2
        self._inventory["iron_sword"] = 1
        self._hp = 9

        # Spawn enemies in other rooms
        self._initial_mob_count = 0
        for room_idx in range(1, len(rooms)):
            r = rooms[room_idx]
            # 1-2 enemies per room
            num = int(self.rng.integers(1, 3))
            for _ in range(num):
                mob_type = "zombie" if self.rng.random() < 0.5 else "skeleton"
                for _att in range(20):
                    mx = int(self.rng.integers(r[0], r[0] + r[2]))
                    my = int(self.rng.integers(r[1], r[1] + r[3]))
                    if (
                        self._world[my][mx] == TILE_DUNGEON_FLOOR
                        and not self._mob_at(mx, my)
                        and (mx, my) != (self._agent_x, self._agent_y)
                    ):
                        _spawn_mob(self._mobs, mob_type, mx, my)
                        self._initial_mob_count += 1
                        break

    # Override movement to use dungeon walkable tiles
    def _handle_move(self, name: str) -> float:
        direction_map = {
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
            and self._world[ny][nx] in _DUNGEON_WALKABLE
            and not self._mob_at(nx, ny)
        ):
            self._agent_x = nx
            self._agent_y = ny
        return 0.0

    def _handle_do(self) -> float:
        old_count = len(self._mobs)
        reward = super()._handle_do()
        new_count = len(self._mobs)
        killed = old_count - new_count
        if killed > 0:
            reward += killed * 5.0
        return reward

    def _subtask_check(
        self, base_reward: float, info: dict[str, Any]
    ) -> tuple[float, bool]:
        hostiles = [m for m in self._mobs if m["type"] != "cow"]
        if not hostiles:
            self._message = "Goal complete! Dungeon floor cleared."
            return 10.0, True
        if self._hp <= 0:
            return -10.0, True
        return 0.0, False

    # Custom rendering for dungeon tiles
    def _render_current_observation(self) -> GridObservation:
        from glyphbench.envs.craftaxfull.classic import _DIR_CHARS, _DIR_NAMES, _MOB_TILES

        half_w = VIEW_WIDTH // 2
        half_h = VIEW_HEIGHT // 2
        grid: list[list[str]] = []
        symbols_seen: set[str] = set()
        facing_ch = _DIR_CHARS.get(self._facing, "@")
        facing_name = _DIR_NAMES.get(self._facing, "right")

        mob_chars: dict[tuple[int, int], str] = {}
        visible_mobs: list[Mob] = []
        for mob in self._mobs:
            mob_chars[(mob["x"], mob["y"])] = _MOB_TILES[mob["type"]]

        for wy in range(VIEW_HEIGHT):
            row: list[str] = []
            for wx in range(VIEW_WIDTH):
                world_x = self._agent_x - half_w + wx
                world_y = self._agent_y - half_h + wy
                if world_x == self._agent_x and world_y == self._agent_y:
                    row.append(facing_ch)
                    symbols_seen.add(facing_ch)
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
                    row.append(TILE_DUNGEON_WALL)
                    symbols_seen.add(TILE_DUNGEON_WALL)
            grid.append(row)

        for mob in self._mobs:
            dx = mob["x"] - self._agent_x
            dy = mob["y"] - self._agent_y
            if abs(dx) <= half_w and abs(dy) <= half_h:
                visible_mobs.append(mob)

        mob_parts = []
        for m in visible_mobs:
            mob_parts.append(f"{m['type']} ({m['hp']}/{m['max_hp']} HP)")
        mob_str = ", ".join(mob_parts) if mob_parts else "none"

        inv_parts = []
        for item, count in sorted(self._inventory.items()):
            if count > 0:
                inv_parts.append(f"{item} x{count}")
        inv_str = ", ".join(inv_parts) if inv_parts else "(empty)"

        hostiles_left = len([m for m in self._mobs if m["type"] != "cow"])

        hud = (
            f"HP: {self._hp}/{self._max_hp}  "
            f"Step: {self._turn} / {self.max_turns}\n"
            f"Facing: {facing_name}\n"
            f"Nearby mobs: {mob_str}\n"
            f"Inventory: {inv_str}\n"
            f"Enemies remaining: {hostiles_left}"
        )

        tile_meanings: dict[str, str] = {
            TILE_DUNGEON_WALL: "dungeon wall (impassable)",
            TILE_DUNGEON_FLOOR: "dungeon floor",
            TILE_ZOMBIE: "zombie (hostile)",
            TILE_SKELETON: "skeleton (hostile)",
        }
        legend_entries: dict[str, str] = {}
        agent_legend = f"you (facing {facing_name})"
        for sym in symbols_seen:
            if sym == facing_ch:
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
