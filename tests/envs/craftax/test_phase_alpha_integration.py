"""Phase-α integration smoke — exercises the layered subsystems together.

Each scenario sets up env state, runs a deterministic action sequence, and
asserts an end-state invariant. These tests catch breakage that unit tests
miss (e.g., a mismatch between spawn-position semantics and the per-step
driver, or between cooldown timing and despawn timing).
"""
from glyphbench.envs.craftax.full import CraftaxFullEnv
from glyphbench.envs.craftax.mechanics.projectiles import (
    ProjectileEntity, ProjectileType,
)


def test_arrow_kills_zombie_on_collision_path() -> None:
    """SHOOT_ARROW + projectile travel + mob HP damage all wired.

    Setup: agent at (5,5) facing east, zombie at (8,5) with 2 HP.
    Attack cooldown=99 prevents zombie from attacking; mob AI still moves
    the zombie toward the player.  After SHOOT_ARROW the arrow is at (6,5);
    the zombie walks to (7,5) (mob AI ran same step).  After one NOOP the
    arrow advances to (7,5) and collides with the zombie (hp 2, damage 2),
    killing it.
    """
    env = CraftaxFullEnv()
    env.reset(seed=42)
    env._agent_x, env._agent_y = 5, 5
    env._facing = (1, 0)
    env._inventory["bow"] = 1
    env._inventory["arrows"] = 5
    env._mobs = []
    # Place a 2-HP zombie 3 tiles east.  High attack_cooldown blocks its
    # attack; it will still chase the player (mob AI moves at dist <= 8).
    env._mobs.append({
        "type": "zombie", "x": 8, "y": 5,
        "hp": 2, "max_hp": 2, "is_boss": False,
        "floor": env._current_floor, "attack_cooldown": 99,
    })

    # Step 1: fire the arrow.  Spawn at (5,5); same-step driver advances
    # to (6,5).  Mob AI moves zombie from (8,5) to (7,5).
    env.step(env.action_spec.names.index("SHOOT_ARROW"))

    # Step 2: NOOP.  Arrow advances (6,5) → (7,5), which is where the
    # zombie now stands.  hit_fn fires: zombie hp 2 - 2 = 0, zombie dies.
    env.step(env.action_spec.names.index("NOOP"))

    surviving = [m for m in env._mobs if m["type"] == "zombie" and m["hp"] > 0]
    assert surviving == [], f"zombie should be dead, found {surviving}"
    # Arrow inventory decremented on the SHOOT_ARROW step.
    assert env._inventory["arrows"] == 4


def test_far_zombie_despawns_after_step() -> None:
    """Mob despawn at dist >= 14 (T23).

    A zombie placed 20 tiles from the player is swept by the despawn pass
    that runs at the end of every _step.  One NOOP is sufficient.
    """
    env = CraftaxFullEnv()
    env.reset(seed=42)
    env._agent_x, env._agent_y = 5, 5
    env._mobs = []
    env._mobs.append({
        "type": "zombie", "x": 25, "y": 5,  # Manhattan dist 20
        "hp": 5, "max_hp": 5, "is_boss": False,
        "floor": env._current_floor, "attack_cooldown": 0,
    })
    env.step(env.action_spec.names.index("NOOP"))
    far_zombies = [m for m in env._mobs if m["x"] == 25 and m["y"] == 5]
    assert far_zombies == [], "far zombie should despawn"


def test_make_arrow_then_shoot_loop() -> None:
    """Action chain: craft 2 arrows at a table, then fire one of them.

    Verifies that MAKE_ARROW (T14) and SHOOT_ARROW (T17) are wired through
    the same _step dispatch table and that the inventory state threads
    correctly across both calls.
    """
    from glyphbench.envs.craftax.base import TILE_TABLE

    env = CraftaxFullEnv()
    env.reset(seed=42)
    env._inventory["wood"] = 4
    env._inventory["stone"] = 4
    env._inventory["bow"] = 1
    env._facing = (1, 0)
    # Place a crafting table adjacent (east) to the agent.
    grid = env._current_grid()
    grid[env._agent_y][env._agent_x + 1] = TILE_TABLE

    env.step(env.action_spec.names.index("MAKE_ARROW"))
    assert env._inventory["arrows"] == 2, (
        f"MAKE_ARROW should yield 2 arrows, got {env._inventory['arrows']}"
    )

    env.step(env.action_spec.names.index("SHOOT_ARROW"))
    assert env._inventory["arrows"] == 1, (
        f"SHOOT_ARROW should consume 1 arrow, got {env._inventory['arrows']}"
    )
    # SHOOT_ARROW handler spawns the projectile; the per-step driver
    # advances it once.  The projectile may have already been consumed by
    # collision or be in flight depending on terrain.  Just assert that the
    # arrow inventory decremented (which proves the action fired).
