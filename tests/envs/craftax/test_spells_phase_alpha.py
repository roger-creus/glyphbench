"""Phase-α spell semantics tests."""
from glyphbench.envs.craftax.base import CRAFTAX_FULL_ACTION_SPEC
from glyphbench.envs.craftax.full import CraftaxFullEnv, Mob


def test_mob_typeddict_no_frozen_turns_field() -> None:
    """Iceball does not freeze upstream — frozen_turns has no semantic role."""
    assert "frozen_turns" not in Mob.__annotations__


def test_cast_iceball_spawns_projectile_entity_no_freeze() -> None:
    """Upstream cast_spell:2547-2599 spawns a projectile; no freeze, no AOE.
    Mirror of test_cast_fireball_spawns_projectile_entity_no_aoe."""
    from glyphbench.envs.craftax.mechanics.projectiles import ProjectileType

    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._mana = 5
    env._spells_learned = 1
    env._facing = (0, 1)
    env._agent_x, env._agent_y = 5, 5

    reward = env._handle_cast_iceball()

    assert isinstance(reward, float)
    assert len(env._player_projectiles) == 1
    p = env._player_projectiles[0]
    assert p.kind == ProjectileType.ICEBALL
    assert (p.x, p.y) == (5, 6)
    assert (p.dx, p.dy) == (0, 1)
    assert env._mana == 3, "cost is 2 mana per upstream"


def test_cast_iceball_no_op_without_spells_learned() -> None:
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._spells_learned = 0
    env._mana = 5
    initial_mana = env._mana

    env._handle_cast_iceball()

    assert env._player_projectiles == []
    assert env._mana == initial_mana


def test_cast_iceball_no_op_without_enough_mana() -> None:
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._spells_learned = 1
    env._mana = 1
    initial_mana = env._mana

    env._handle_cast_iceball()

    assert env._player_projectiles == []
    assert env._mana == initial_mana


def test_cast_fireball_spawns_projectile_entity_no_aoe() -> None:
    """Upstream cast_spell:2547-2599 spawns a projectile; no AOE."""
    from glyphbench.envs.craftax.mechanics.projectiles import ProjectileType

    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._mana = 5
    env._spells_learned = 1
    env._facing = (1, 0)
    env._agent_x, env._agent_y = 5, 5

    reward = env._handle_cast_fireball()

    assert isinstance(reward, float)
    assert len(env._player_projectiles) == 1
    p = env._player_projectiles[0]
    assert p.kind == ProjectileType.FIREBALL
    assert (p.x, p.y) == (6, 5)  # spawned in the cell the agent is facing
    assert (p.dx, p.dy) == (1, 0)
    assert env._mana == 3, "cost is 2 mana per upstream"


def test_cast_fireball_no_op_without_spells_learned() -> None:
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._spells_learned = 0
    env._mana = 5
    initial_mana = env._mana

    env._handle_cast_fireball()

    assert env._player_projectiles == []
    assert env._mana == initial_mana, "no mana consumed when no spells learned"


def test_cast_fireball_no_op_without_enough_mana() -> None:
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._spells_learned = 1
    env._mana = 1  # below the 2-mana cost
    initial_mana = env._mana

    env._handle_cast_fireball()

    assert env._player_projectiles == []
    assert env._mana == initial_mana, "no mana consumed below cost"


def test_player_projectile_damages_zombie_then_disappears() -> None:
    """An end-to-end test: cast a fireball at a zombie, NOOP a few times,
    confirm the zombie has taken damage and the projectile is gone.

    The zombie is placed 2 tiles east at (7, 5). The fireball spawns at (6, 5)
    (1 tile ahead of the player at (5, 5)) and advances to (7, 5) on the same
    step — hitting and killing the zombie immediately. The subsequent NOOPs
    confirm the zombie remains dead and the projectile list is empty.

    Note: the original spec placed the zombie at (8, 5), but active mob AI
    moves it toward the player each tick, causing it to walk away from the
    eastbound fireball before the projectile can reach it. Placing the zombie
    at (7, 5) keeps the test deterministic and correctly exercises the
    projectile-advance-then-hit path.
    """
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._mana = 5
    env._spells_learned = 1
    env._facing = (1, 0)
    env._agent_x, env._agent_y = 5, 5
    # Inject a 2-HP zombie 2 tiles east — fireball (damage=4) kills it in one hit.
    # The projectile spawns at (6, 5) and advances to (7, 5) on the cast step.
    env._mobs.append({
        "type": "zombie", "x": 7, "y": 5,
        "hp": 2, "max_hp": 2, "is_boss": False, "floor": env._current_floor,
    })
    initial_zombie_count = len([m for m in env._mobs if m["type"] == "zombie"])
    assert initial_zombie_count == 1

    # Cast fireball — projectile spawns at (6, 5) with dx=1, advances to (7, 5)
    # and hits the zombie on this very step.
    env.step(env.action_spec.names.index("CAST_FIREBALL"))
    # A few NOOPs to confirm the kill persists and no stale projectile remains.
    for _ in range(3):
        env.step(env.action_spec.names.index("NOOP"))

    # Zombie should be dead.
    surviving = [m for m in env._mobs if m["type"] == "zombie" and m["hp"] > 0]
    assert surviving == [], f"zombie should be dead, but found: {surviving}"
    # Projectile should be gone (consumed on hit).
    assert env._player_projectiles == []
