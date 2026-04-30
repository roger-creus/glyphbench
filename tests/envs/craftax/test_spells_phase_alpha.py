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
    env._learned_spells["fireball"] = True
    env._learned_spells["iceball"] = True
    env._facing = (0, 1)
    env._agent_x, env._agent_y = 5, 5

    reward = env._handle_cast_iceball()

    assert isinstance(reward, float)
    assert len(env._player_projectiles) == 1
    p = env._player_projectiles[0]
    assert p.kind == ProjectileType.ICEBALL
    assert (p.x, p.y) == (5, 5)
    assert (p.dx, p.dy) == (0, 1)
    assert env._mana == 3, "cost is 2 mana per upstream"


def test_cast_iceball_no_op_without_spells_learned() -> None:
    env = CraftaxFullEnv()
    env.reset(seed=0)
    # After reset, _learned_spells is all False — no need to set
    env._mana = 5
    initial_mana = env._mana

    env._handle_cast_iceball()

    assert env._player_projectiles == []
    assert env._mana == initial_mana


def test_cast_iceball_no_op_without_enough_mana() -> None:
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._learned_spells["iceball"] = True
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
    env._learned_spells["fireball"] = True
    env._facing = (1, 0)
    env._agent_x, env._agent_y = 5, 5

    reward = env._handle_cast_fireball()

    assert isinstance(reward, float)
    assert len(env._player_projectiles) == 1
    p = env._player_projectiles[0]
    assert p.kind == ProjectileType.FIREBALL
    assert (p.x, p.y) == (5, 5)  # spawned at player tile; same-step advance carries it to (6,5)
    assert (p.dx, p.dy) == (1, 0)
    assert env._mana == 3, "cost is 2 mana per upstream"


def test_cast_fireball_no_op_without_spells_learned() -> None:
    env = CraftaxFullEnv()
    env.reset(seed=0)
    # After reset, _learned_spells is all False — no need to set
    env._mana = 5
    initial_mana = env._mana

    env._handle_cast_fireball()

    assert env._player_projectiles == []
    assert env._mana == initial_mana, "no mana consumed when no spells learned"


def test_cast_fireball_no_op_without_enough_mana() -> None:
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._learned_spells["fireball"] = True
    env._mana = 1  # below the 2-mana cost
    initial_mana = env._mana

    env._handle_cast_fireball()

    assert env._player_projectiles == []
    assert env._mana == initial_mana, "no mana consumed below cost"


def test_player_projectile_damages_adjacent_zombie_on_cast_step() -> None:
    """Upstream-faithful spawn: fireball placed at player tile, same-step
    advance carries it to player_pos + 1*dir. A zombie at player + 1*dir
    (immediately in front of the agent) must be hit on the cast step itself —
    the bug this test guards against was the old player+dir spawn which caused
    the projectile to advance to player+2*dir on the cast step, skipping the
    adjacent mob entirely.

    Setup: player at (5, 5) facing east (1, 0), zombie at (6, 5) with 2 HP.
    Fireball spawns at (5, 5), advances to (6, 5), hits zombie (damage=4 > 2 HP).
    Post-cast: zombie dead, projectile consumed.
    """
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._mana = 5
    env._learned_spells["fireball"] = True
    env._facing = (1, 0)
    env._agent_x, env._agent_y = 5, 5
    env._mobs.append({
        "type": "zombie", "x": 6, "y": 5,
        "hp": 2, "max_hp": 2, "is_boss": False, "floor": env._current_floor,
        "attack_cooldown": 0,
    })

    env.step(env.action_spec.names.index("CAST_FIREBALL"))

    # No zombies should survive (killed on the cast step).
    assert not [m for m in env._mobs if m["type"] == "zombie"]
    # Projectile consumed on hit — list must be empty.
    assert env._player_projectiles == []
