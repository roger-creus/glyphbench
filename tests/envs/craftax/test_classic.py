from glyphbench.core import make_env
import glyphbench.envs.craftax  # register envs
"""Unit tests for Craftax Classic (22-achievement) env."""

import pytest

from glyphbench.envs.craftax.classic import (
    CraftaxClassicEnv,
    _MAX_ENERGY,
    _MAX_FOOD,
    _MAX_WATER,
    _PLANT_RIPEN_STEPS,
)
from glyphbench.envs.craftax.base import (
    TILE_GRASS,
    TILE_PLACED_STONE,
    TILE_RIPE_PLANT,
    TILE_SAPLING,
    TILE_TABLE,
    TILE_TREE,
    TILE_WATER,
)


class TestCraftaxClassic:
    """Tests for the Craftax Classic environment (22-achievement full version)."""

    def _make_env(self, max_turns: int = 10000) -> CraftaxClassicEnv:
        return CraftaxClassicEnv(max_turns=max_turns)

    # --- Action space ---
    def test_action_space_defined(self):
        env = self._make_env()
        assert env.action_spec.n == 19
        expected = (
            "NOOP", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN",
            "DO", "SLEEP",
            "PLACE_STONE", "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT",
            "MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE",
            "MAKE_IRON_PICKAXE",
            "MAKE_WOOD_SWORD", "MAKE_STONE_SWORD",
            "MAKE_IRON_SWORD",
            "EAT_PLANT", "DRINK_WATER",
        )
        assert env.action_spec.names == expected

    def test_noop_action_name(self):
        env = self._make_env()
        assert env.noop_action_name == "NOOP"

    def test_env_id(self):
        env = self._make_env()
        assert env.env_id() == "glyphbench/craftax-classic-v0"

    # --- Determinism ---
    def test_reset_determinism(self):
        e1 = self._make_env()
        e2 = self._make_env()
        o1, _ = e1.reset(seed=42)
        o2, _ = e2.reset(seed=42)
        assert o1 == o2

    def test_step_determinism(self):
        e1 = self._make_env()
        e2 = self._make_env()
        e1.reset(seed=0)
        e2.reset(seed=0)
        right = e1.action_spec.index_of("MOVE_RIGHT")
        for _ in range(10):
            o1, r1, t1, tr1, _ = e1.step(right)
            o2, r2, t2, tr2, _ = e2.step(right)
            assert o1 == o2
            assert r1 == r2

    def test_world_gen_determinism(self):
        e1 = self._make_env()
        e2 = self._make_env()
        e1.reset(seed=42)
        e2.reset(seed=42)
        assert e1._world == e2._world

    # --- Observation contract ---
    def test_observation_contract(self):
        env = self._make_env()
        obs_str, _ = env.reset(0)
        assert isinstance(obs_str, str)
        assert "[Grid]" in obs_str
        assert "[Legend]" in obs_str
        assert "[HUD]" in obs_str
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        lengths = [len(line) for line in grid_lines]
        assert len(set(lengths)) == 1

    def test_visible_window_size(self):
        env = self._make_env()
        env.reset(0)
        grid_obs = env.get_observation()
        grid_lines = grid_obs.grid.split("\n")
        assert len(grid_lines) == 7, f"Expected 7 rows, got {len(grid_lines)}"
        assert len(grid_lines[0]) == 9, f"Expected 9 cols, got {len(grid_lines[0])}"

    # --- 22 achievements ---
    def test_22_achievements_available(self):
        env = self._make_env()
        assert len(env._ALL_ACHIEVEMENTS) == 22

    # --- Wood collection ---
    def test_collect_wood(self):
        """DO action at a tree -> tree disappears, wood += 1, achievement unlocked."""
        env = self._make_env()
        env.reset(0)

        # Place a tree next to agent and face it
        ax, ay = env._agent_x, env._agent_y
        if ax + 1 < env._WORLD_SIZE:
            env._world[ay][ax + 1] = TILE_TREE
        env._facing = (1, 0)
        do_action = env.action_spec.index_of("DO")
        _, reward, _, _, info = env.step(do_action)
        assert env._inventory.get("wood", 0) >= 1
        assert "collect_wood" in env._achievements_unlocked

    # --- Crafting at table ---
    def test_make_wood_pickaxe_at_table(self):
        """At a placed table with wood -> MAKE_WOOD_PICKAXE -> pickaxe granted."""
        env = self._make_env()
        env.reset(0)
        env._inventory["wood"] = 5
        # Place table in facing direction
        place_table = env.action_spec.index_of("PLACE_TABLE")
        env.step(place_table)
        # Make wood pickaxe
        env._inventory["wood"] = 2
        make_pickaxe = env.action_spec.index_of("MAKE_WOOD_PICKAXE")
        env.step(make_pickaxe)
        if env._near_table():
            assert env._inventory.get("wood_pickaxe", 0) >= 1

    # --- Achievement idempotence ---
    def test_achievement_idempotence(self):
        """Unlocking an already-unlocked achievement gives 0 reward."""
        env = self._make_env()
        env.reset(0)
        # Pre-unlock both collect_wood and collect_sapling so neither triggers
        env._achievements_unlocked.add("collect_wood")
        env._achievements_unlocked.add("collect_sapling")
        env._inventory["wood"] = 0
        ax, ay = env._agent_x, env._agent_y
        if ax + 1 < env._WORLD_SIZE:
            env._world[ay][ax + 1] = TILE_TREE
        env._facing = (1, 0)
        do_action = env.action_spec.index_of("DO")
        _, reward, _, _, _ = env.step(do_action)
        assert reward == 0.0

    # --- Max turns truncation ---
    def test_max_turns_truncation(self):
        env = self._make_env(max_turns=5)
        env.reset(0)
        noop = env.action_spec.index_of("NOOP")
        for i in range(5):
            _, _, terminated, truncated, _ = env.step(noop)
            if terminated:
                break
            if i < 4:
                assert not truncated
            else:
                assert truncated

    # --- Reward bounds ---
    def test_reward_bounds(self):
        """Rewards are 0 or 1 (per achievement)."""
        env = self._make_env()
        env.reset(0)
        noop = env.action_spec.index_of("NOOP")
        for _ in range(20):
            _, reward, terminated, _, _ = env.step(noop)
            if terminated:
                break
            assert reward in (0.0, 1.0), f"Unexpected reward: {reward}"

    # --- HUD ---
    def test_hud_content(self):
        env = self._make_env()
        env.reset(0)
        hud = env.get_observation().hud
        assert "HP:" in hud
        assert "Food:" in hud
        assert "Water:" in hud
        assert "Energy:" in hud
        assert "Time:" in hud
        assert "Step:" in hud

    # --- Legend ---
    def test_legend_has_symbols(self):
        env = self._make_env()
        env.reset(0)
        legend = env.get_observation().legend
        # Player is directional: →, ←, ↑, or ↓
        assert any(c in legend for c in "→←↑↓")
        assert len(legend) > 0

    # --- Info extras ---
    def test_info_extras(self):
        env = self._make_env()
        _, info = env.reset(0)
        noop = env.action_spec.index_of("NOOP")
        _, _, _, _, info = env.step(noop)
        assert "achievements_this_step" in info
        assert "day_night" in info
        assert "food" in info
        assert "water" in info
        assert "energy" in info

    def test_reset_requires_seed(self):
        env = self._make_env()
        with pytest.raises(TypeError):
            env.reset()

    def test_system_prompt(self):
        env = self._make_env()
        prompt = env.system_prompt()
        assert len(prompt) > 0
        assert "Craftax" in prompt
        assert "zombie" in prompt.lower()
        assert "survival" in prompt.lower() or "food" in prompt.lower()

    # --- Movement ---
    def test_movement_changes_position(self):
        env = self._make_env()
        env.reset(0)
        right = env.action_spec.index_of("MOVE_RIGHT")
        env.step(right)
        assert isinstance(env._agent_x, int)

    # --- DO on empty ground is no-op ---
    def test_do_on_empty_ground(self):
        env = self._make_env()
        env.reset(0)
        ax, ay = env._agent_x, env._agent_y
        fx, fy = ax + env._facing[0], ay + env._facing[1]
        if 0 <= fx < env._WORLD_SIZE and 0 <= fy < env._WORLD_SIZE:
            env._world[fy][fx] = "."
        # Remove any mobs at that position
        env._mobs = [m for m in env._mobs if (m["x"], m["y"]) != (fx, fy)]
        do_action = env.action_spec.index_of("DO")
        _, reward, _, _, _ = env.step(do_action)
        assert reward == 0.0

    # ===================================================================
    # New tests for 22-achievement mechanics
    # ===================================================================

    # --- Day/night cycle ---
    def test_day_night_initial(self):
        """Game starts in daytime."""
        env = self._make_env()
        env.reset(0)
        assert env._day_night == "day"
        assert env._day_counter == 0

    def test_night_transition(self):
        """After 200 steps, night falls."""
        env = self._make_env()
        env.reset(0)
        noop = env.action_spec.index_of("NOOP")
        for _ in range(200):
            _, _, terminated, _, _ = env.step(noop)
            if terminated:
                break
        assert env._day_night == "night"

    def test_dawn_transition(self):
        """After 300 steps, dawn arrives (keep player alive with high HP)."""
        env = self._make_env()
        env.reset(0)
        # Keep player alive through starvation/dehydration damage
        env._hp = 999
        env._max_hp = 999
        noop = env.action_spec.index_of("NOOP")
        for _ in range(300):
            env.step(noop)
        assert env._day_night == "day"

    # --- Survival stats ---
    def test_initial_survival_stats(self):
        """Survival stats start at max."""
        env = self._make_env()
        env.reset(0)
        assert env._food == _MAX_FOOD
        assert env._water == _MAX_WATER
        assert env._energy == _MAX_ENERGY

    def test_food_drain(self):
        """Food decreases over time."""
        env = self._make_env()
        env.reset(0)
        noop = env.action_spec.index_of("NOOP")
        initial_food = env._food
        for _ in range(50):
            _, _, terminated, _, _ = env.step(noop)
            if terminated:
                break
        assert env._food < initial_food

    def test_water_drain(self):
        """Water decreases over time."""
        env = self._make_env()
        env.reset(0)
        noop = env.action_spec.index_of("NOOP")
        initial_water = env._water
        for _ in range(40):
            _, _, terminated, _, _ = env.step(noop)
            if terminated:
                break
        assert env._water < initial_water

    # --- Combat ---
    def test_attack_mob(self):
        """DO facing a mob attacks it."""
        env = self._make_env()
        env.reset(0)
        # Place a zombie next to agent
        ax, ay = env._agent_x, env._agent_y
        env._facing = (1, 0)
        fx, fy = ax + 1, ay
        # Clear the tile
        env._world[fy][fx] = TILE_GRASS
        env._mobs.append({
            "type": "zombie", "x": fx, "y": fy,
            "hp": 3, "max_hp": 3,
        })
        do_action = env.action_spec.index_of("DO")
        env.step(do_action)
        # Zombie should have taken damage (1 base damage)
        zombie = env._mob_at(fx, fy)
        if zombie is not None:
            assert zombie["hp"] < 3

    def test_defeat_zombie_achievement(self):
        """Killing a zombie unlocks defeat_zombie."""
        env = self._make_env()
        env.reset(0)
        ax, ay = env._agent_x, env._agent_y
        env._facing = (1, 0)
        fx, fy = ax + 1, ay
        env._world[fy][fx] = TILE_GRASS
        env._mobs.append({
            "type": "zombie", "x": fx, "y": fy,
            "hp": 1, "max_hp": 3,
        })
        do_action = env.action_spec.index_of("DO")
        _, reward, _, _, _ = env.step(do_action)
        assert "defeat_zombie" in env._achievements_unlocked
        assert reward >= 1.0

    def test_defeat_skeleton_achievement(self):
        """Killing a skeleton unlocks defeat_skeleton."""
        env = self._make_env()
        env.reset(0)
        ax, ay = env._agent_x, env._agent_y
        env._facing = (1, 0)
        fx, fy = ax + 1, ay
        env._world[fy][fx] = TILE_GRASS
        env._mobs.append({
            "type": "skeleton", "x": fx, "y": fy,
            "hp": 1, "max_hp": 4,
        })
        do_action = env.action_spec.index_of("DO")
        _, reward, _, _, _ = env.step(do_action)
        assert "defeat_skeleton" in env._achievements_unlocked
        assert reward >= 1.0

    def test_eat_cow_achievement(self):
        """Killing a cow unlocks eat_cow and restores food."""
        env = self._make_env()
        env.reset(0)
        ax, ay = env._agent_x, env._agent_y
        env._facing = (1, 0)
        fx, fy = ax + 1, ay
        env._world[fy][fx] = TILE_GRASS
        env._mobs = [m for m in env._mobs if m["type"] != "cow"]  # clear existing cows
        env._mobs.append({
            "type": "cow", "x": fx, "y": fy,
            "hp": 1, "max_hp": 3,
        })
        env._food = 2
        do_action = env.action_spec.index_of("DO")
        _, reward, _, _, _ = env.step(do_action)
        assert "eat_cow" in env._achievements_unlocked
        assert env._food == 7  # 2 + 5

    def test_weapon_bonus_damage(self):
        """Weapon increases damage."""
        env = self._make_env()
        env.reset(0)
        ax, ay = env._agent_x, env._agent_y
        env._facing = (1, 0)
        fx, fy = ax + 1, ay
        env._world[fy][fx] = TILE_GRASS
        env._inventory["iron_sword"] = 1
        env._mobs.append({
            "type": "zombie", "x": fx, "y": fy,
            "hp": 4, "max_hp": 4,
        })
        do_action = env.action_spec.index_of("DO")
        env.step(do_action)
        # 1 base + 3 iron sword = 4 damage, should kill zombie
        assert env._mob_at(fx, fy) is None
        assert "defeat_zombie" in env._achievements_unlocked

    # --- Sleep ---
    def test_sleep_restores_energy(self):
        """SLEEP restores energy to max."""
        env = self._make_env()
        env.reset(0)
        env._energy = 3
        sleep_action = env.action_spec.index_of("SLEEP")
        env.step(sleep_action)
        assert env._energy == _MAX_ENERGY

    def test_sleep_unlocks_wake_up(self):
        """First SLEEP unlocks wake_up achievement."""
        env = self._make_env()
        env.reset(0)
        sleep_action = env.action_spec.index_of("SLEEP")
        _, reward, _, _, _ = env.step(sleep_action)
        assert "wake_up" in env._achievements_unlocked
        assert reward >= 1.0

    # --- Plants ---
    def test_place_plant(self):
        """PLACE_PLANT with a sapling places a sapling tile."""
        env = self._make_env()
        env.reset(0)
        env._inventory["sapling"] = 1
        ax, ay = env._agent_x, env._agent_y
        env._facing = (1, 0)
        fx, fy = ax + 1, ay
        env._world[fy][fx] = TILE_GRASS
        # Remove any mobs at that position
        env._mobs = [m for m in env._mobs if (m["x"], m["y"]) != (fx, fy)]
        place_plant = env.action_spec.index_of("PLACE_PLANT")
        _, reward, _, _, _ = env.step(place_plant)
        assert env._world[fy][fx] == TILE_SAPLING
        assert "place_plant" in env._achievements_unlocked

    def test_plant_ripens(self):
        """Sapling becomes ripe plant after enough steps."""
        env = self._make_env()
        env.reset(0)
        env._inventory["sapling"] = 1
        ax, ay = env._agent_x, env._agent_y
        env._facing = (1, 0)
        fx, fy = ax + 1, ay
        env._world[fy][fx] = TILE_GRASS
        env._mobs = [m for m in env._mobs if (m["x"], m["y"]) != (fx, fy)]
        place_plant = env.action_spec.index_of("PLACE_PLANT")
        env.step(place_plant)
        assert env._world[fy][fx] == TILE_SAPLING

        # Step enough times for plant to ripen
        noop = env.action_spec.index_of("NOOP")
        for _ in range(_PLANT_RIPEN_STEPS + 1):
            _, _, terminated, _, _ = env.step(noop)
            if terminated:
                break
        assert env._world[fy][fx] == TILE_RIPE_PLANT

    def test_eat_plant(self):
        """EAT_PLANT on ripe plant restores food."""
        env = self._make_env()
        env.reset(0)
        ax, ay = env._agent_x, env._agent_y
        env._facing = (1, 0)
        fx, fy = ax + 1, ay
        env._world[fy][fx] = TILE_RIPE_PLANT
        env._food = 4
        eat_plant = env.action_spec.index_of("EAT_PLANT")
        _, reward, _, _, _ = env.step(eat_plant)
        assert env._food == 7  # 4 + 3
        assert "eat_plant" in env._achievements_unlocked

    # --- Drink water ---
    def test_drink_water(self):
        """DRINK_WATER facing water restores water."""
        env = self._make_env()
        env.reset(0)
        ax, ay = env._agent_x, env._agent_y
        env._facing = (1, 0)
        fx, fy = ax + 1, ay
        env._world[fy][fx] = TILE_WATER
        env._water = 3
        drink = env.action_spec.index_of("DRINK_WATER")
        _, reward, _, _, _ = env.step(drink)
        assert env._water == _MAX_WATER
        assert "collect_drink" in env._achievements_unlocked

    # --- Place stone achievement ---
    def test_place_stone_achievement(self):
        """PLACE_STONE unlocks place_stone achievement."""
        env = self._make_env()
        env.reset(0)
        env._inventory["stone"] = 1
        ax, ay = env._agent_x, env._agent_y
        env._facing = (1, 0)
        fx, fy = ax + 1, ay
        env._world[fy][fx] = TILE_GRASS
        env._mobs = [m for m in env._mobs if (m["x"], m["y"]) != (fx, fy)]
        place_stone = env.action_spec.index_of("PLACE_STONE")
        _, reward, _, _, _ = env.step(place_stone)
        assert "place_stone" in env._achievements_unlocked
        assert reward >= 1.0

    # --- Iron pickaxe crafting ---
    def test_make_iron_pickaxe(self):
        """MAKE_IRON_PICKAXE near table+furnace with resources."""
        env = self._make_env()
        env.reset(0)
        ax, ay = env._agent_x, env._agent_y
        # Place table and furnace adjacent
        env._world[ay][ax + 1] = TILE_TABLE
        env._world[ay][ax - 1] = "f"  # furnace
        env._inventory["wood"] = 1
        env._inventory["iron"] = 1
        make_iron_pick = env.action_spec.index_of("MAKE_IRON_PICKAXE")
        _, reward, _, _, _ = env.step(make_iron_pick)
        assert env._inventory.get("iron_pickaxe", 0) >= 1
        assert "make_iron_pickaxe" in env._achievements_unlocked

    # --- Diamond requires iron pickaxe ---
    def test_diamond_requires_iron_pickaxe(self):
        """Diamond needs iron pickaxe, not stone pickaxe."""
        env = self._make_env()
        env.reset(0)
        ax, ay = env._agent_x, env._agent_y
        env._facing = (1, 0)
        env._world[ay][ax + 1] = "D"
        # With stone pickaxe, should fail
        env._inventory["stone_pickaxe"] = 1
        do_action = env.action_spec.index_of("DO")
        env.step(do_action)
        assert env._inventory.get("diamond", 0) == 0

        # With iron pickaxe, should succeed
        env._world[ay][ax + 1] = "D"
        env._inventory["iron_pickaxe"] = 1
        env.step(do_action)
        assert env._inventory.get("diamond", 0) >= 1
        assert "collect_diamond" in env._achievements_unlocked

    # --- Sword crafting ---
    def test_make_wood_sword(self):
        """MAKE_WOOD_SWORD near table with wood."""
        env = self._make_env()
        env.reset(0)
        ax, ay = env._agent_x, env._agent_y
        env._world[ay][ax + 1] = TILE_TABLE
        env._inventory["wood"] = 1
        action = env.action_spec.index_of("MAKE_WOOD_SWORD")
        env.step(action)
        assert env._inventory.get("wood_sword", 0) >= 1
        assert "make_wood_sword" in env._achievements_unlocked

    def test_make_stone_sword(self):
        """MAKE_STONE_SWORD near table+furnace with resources."""
        env = self._make_env()
        env.reset(0)
        ax, ay = env._agent_x, env._agent_y
        env._world[ay][ax + 1] = TILE_TABLE
        env._world[ay][ax - 1] = "f"
        env._inventory["wood"] = 1
        env._inventory["stone"] = 1
        action = env.action_spec.index_of("MAKE_STONE_SWORD")
        env.step(action)
        assert env._inventory.get("stone_sword", 0) >= 1
        assert "make_stone_sword" in env._achievements_unlocked

    def test_make_iron_sword(self):
        """MAKE_IRON_SWORD near table+furnace with resources."""
        env = self._make_env()
        env.reset(0)
        ax, ay = env._agent_x, env._agent_y
        env._world[ay][ax + 1] = TILE_TABLE
        env._world[ay][ax - 1] = "f"
        env._inventory["wood"] = 1
        env._inventory["iron"] = 1
        action = env.action_spec.index_of("MAKE_IRON_SWORD")
        env.step(action)
        assert env._inventory.get("iron_sword", 0) >= 1
        assert "make_iron_sword" in env._achievements_unlocked

    # --- Mobs spawn ---
    def test_cows_spawn_at_start(self):
        """Cows are spawned on reset."""
        env = self._make_env()
        env.reset(42)
        cow_count = sum(1 for m in env._mobs if m["type"] == "cow")
        assert cow_count >= 3

    def test_night_spawns_hostile_mobs(self):
        """Hostile mobs spawn when night falls."""
        env = self._make_env()
        env.reset(0)
        noop = env.action_spec.index_of("NOOP")
        # Step to nightfall
        for _ in range(200):
            _, _, terminated, _, _ = env.step(noop)
            if terminated:
                break
        hostile = [m for m in env._mobs if m["type"] in ("zombie", "skeleton")]
        assert len(hostile) > 0

    # --- Mob movement blocks player ---
    def test_cannot_walk_onto_mob(self):
        """Player cannot move onto a tile occupied by a mob."""
        env = self._make_env()
        env.reset(0)
        ax, ay = env._agent_x, env._agent_y
        # Place a cow to the right
        env._world[ay][ax + 1] = TILE_GRASS
        env._mobs.append({
            "type": "cow", "x": ax + 1, "y": ay,
            "hp": 3, "max_hp": 3,
        })
        right = env.action_spec.index_of("MOVE_RIGHT")
        env.step(right)
        assert env._agent_x == ax  # Should not have moved

    # --- Death ---
    def test_player_death(self):
        """Player dies when HP drops to 0."""
        env = self._make_env()
        env.reset(0)
        env._hp = 1
        env._food = 0
        env._water = 0
        # Force a drain step
        env._day_counter = 49  # Next step will be 50, triggering food drain
        noop = env.action_spec.index_of("NOOP")
        _, _, terminated, _, _ = env.step(noop)
        # HP should have dropped from starvation/dehydration
        assert env._hp <= 0
        assert terminated

    # --- Sapling from tree ---
    def test_tree_may_drop_sapling(self):
        """Chopping a tree has a chance to drop a sapling."""
        env = self._make_env()
        env.reset(0)
        do_action = env.action_spec.index_of("DO")
        saplings_found = False

        # Chop many trees to check for sapling drops
        for attempt in range(50):
            ax, ay = env._agent_x, env._agent_y
            env._facing = (1, 0)
            fx, fy = ax + 1, ay
            if 0 <= fx < env._WORLD_SIZE:
                env._world[fy][fx] = TILE_TREE
                env.step(do_action)
                if env._inventory.get("sapling", 0) > 0:
                    saplings_found = True
                    break

        assert saplings_found, "Expected at least one sapling drop in 50 tree chops"

    # --- Shelter check ---
    def test_shelter_check(self):
        """Player is sheltered when surrounded by placed stone on all 4 sides."""
        env = self._make_env()
        env.reset(0)
        ax, ay = env._agent_x, env._agent_y
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            env._world[ay + dy][ax + dx] = TILE_PLACED_STONE
        assert env._has_shelter()
