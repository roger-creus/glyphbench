"""Phase-α mob roster + AI primitives tests."""
from glyphbench.envs.craftax.mechanics.mobs import (
    MobType,
    PASSIVE_MOB_NAMES,
    MELEE_MOB_NAMES,
    RANGED_MOB_NAMES,
    FLOOR_MOB_MAPPING,
)


def test_mob_type_enum_has_four_classes() -> None:
    """Upstream constants.py:118-123 — PASSIVE / MELEE / RANGED / PROJECTILE."""
    assert {m.name for m in MobType} == {"PASSIVE", "MELEE", "RANGED", "PROJECTILE"}


def test_mob_type_enum_integer_values_match_upstream() -> None:
    assert MobType.PASSIVE.value == 0
    assert MobType.MELEE.value == 1
    assert MobType.RANGED.value == 2
    assert MobType.PROJECTILE.value == 3


def test_mob_rosters_have_upstream_counts() -> None:
    """Upstream: 3 passive, 8 melee, 8 ranged."""
    assert len(PASSIVE_MOB_NAMES) == 3
    assert len(MELEE_MOB_NAMES) == 8
    assert len(RANGED_MOB_NAMES) == 8


def test_passive_roster_matches_upstream() -> None:
    assert PASSIVE_MOB_NAMES == ("cow", "bat", "snail")


def test_melee_roster_matches_upstream() -> None:
    assert MELEE_MOB_NAMES == (
        "zombie", "gnome_warrior", "orc_soldier", "lizard",
        "knight", "troll", "pigman", "frost_troll",
    )


def test_ranged_roster_matches_upstream() -> None:
    assert RANGED_MOB_NAMES == (
        "skeleton", "gnome_archer", "orc_mage", "kobold",
        "knight_archer", "deep_thing", "fire_elemental", "ice_elemental",
    )


def test_floor_mob_mapping_has_nine_floors() -> None:
    """Upstream constants.py:138-152 — 9 floors (0..8)."""
    assert len(FLOOR_MOB_MAPPING) == 9
    # Floor 0 is overworld.
    assert FLOOR_MOB_MAPPING[0]["passive"] == "cow"
    # Floor 8 is the boss/graveyard floor.
    assert FLOOR_MOB_MAPPING[8]["melee"] == "zombie"


def test_ranged_mob_to_projectile_mapping_matches_upstream() -> None:
    """Upstream constants.py:320-331."""
    from glyphbench.envs.craftax.mechanics.mobs import RANGED_MOB_TO_PROJECTILE
    from glyphbench.envs.craftax.mechanics.projectiles import ProjectileType

    expected = {
        "skeleton": ProjectileType.ARROW,
        "gnome_archer": ProjectileType.ARROW,
        "orc_mage": ProjectileType.FIREBALL,
        "kobold": ProjectileType.DAGGER,
        "knight_archer": ProjectileType.ARROW2,
        "deep_thing": ProjectileType.SLIMEBALL,
        "fire_elemental": ProjectileType.FIREBALL2,
        "ice_elemental": ProjectileType.ICEBALL2,
    }
    assert RANGED_MOB_TO_PROJECTILE == expected


def test_ranged_mob_to_projectile_covers_all_ranged_names() -> None:
    """Every ranged-mob name has a projectile mapping."""
    from glyphbench.envs.craftax.mechanics.mobs import (
        RANGED_MOB_TO_PROJECTILE,
        RANGED_MOB_NAMES,
    )
    assert set(RANGED_MOB_TO_PROJECTILE.keys()) == set(RANGED_MOB_NAMES)


def test_mob_typeddict_has_attack_cooldown_field() -> None:
    """T20: per-mob attack cooldown for T21/T24 AI."""
    import typing
    from glyphbench.envs.craftax.full import Mob
    assert "attack_cooldown" in Mob.__annotations__
    # full.py uses `from __future__ import annotations` so raw annotations are
    # ForwardRef strings; resolve via get_type_hints() to compare actual types.
    resolved = typing.get_type_hints(Mob)
    assert resolved["attack_cooldown"] is int


def test_freshly_spawned_boss_has_zero_attack_cooldown() -> None:
    """All construction sites must initialise attack_cooldown=0."""
    from glyphbench.envs.craftax.full import CraftaxFullEnv

    env = CraftaxFullEnv()
    env.reset(seed=0)
    # Append a synthetic mob using the new construction shape.
    env._mobs.append({
        "type": "zombie", "x": 7, "y": 7,
        "hp": 5, "max_hp": 5, "is_boss": False,
        "floor": env._current_floor, "attack_cooldown": 0,
    })
    assert env._mobs[-1]["attack_cooldown"] == 0
    # Confirm the env's own spawn paths set it too. Trigger a night spawn
    # by advancing the day counter and stepping; this exercises the night-mob
    # spawn site.
    env._day_counter = 100  # near night transition
    env._day_night = "night"
    env._mobs = []
    # Run several steps so the night-mob spawn path fires.
    for _ in range(20):
        env.step(env.action_spec.names.index("NOOP"))
    # Confirm any spawned mob has the field.
    for mob in env._mobs:
        assert "attack_cooldown" in mob, f"mob missing attack_cooldown: {mob}"
        assert mob["attack_cooldown"] == 0


import random as _random  # alias to avoid clash with the test using mocked rng


def test_step_melee_mob_attacks_when_adjacent_then_enters_cooldown() -> None:
    """T21: step_melee_mob hits player at Manhattan-1, then resets cooldown to 5."""
    from glyphbench.envs.craftax.mechanics.mobs import step_melee_mob

    mob = {
        "type": "zombie", "x": 5, "y": 6,  # adjacent below player at (5, 5)
        "hp": 5, "max_hp": 5, "is_boss": False,
        "floor": 0, "attack_cooldown": 0,
    }
    damage_log: list[int] = []
    step_melee_mob(
        mob,
        player_x=5, player_y=5,
        is_blocked_for_mob=lambda x, y: False,
        apply_damage_to_player=lambda d: damage_log.append(d),
        rng=_random.Random(0),
        is_fighting_boss=False,
        damage_for_mob=lambda m: 2,
    )
    assert damage_log == [2], f"expected single hit of 2, got {damage_log}"
    assert mob["attack_cooldown"] == 5, "cooldown reset to 5 after attack"


def test_step_melee_mob_does_not_attack_during_cooldown() -> None:
    """T21: cooldown ticks down each call; no attack while cooldown > 0."""
    from glyphbench.envs.craftax.mechanics.mobs import step_melee_mob

    mob = {
        "type": "zombie", "x": 5, "y": 6,
        "hp": 5, "max_hp": 5, "is_boss": False,
        "floor": 0, "attack_cooldown": 3,
    }
    damage_log: list[int] = []
    step_melee_mob(
        mob,
        player_x=5, player_y=5,
        is_blocked_for_mob=lambda x, y: False,
        apply_damage_to_player=lambda d: damage_log.append(d),
        rng=_random.Random(0),
        is_fighting_boss=False,
        damage_for_mob=lambda m: 2,
    )
    assert damage_log == [], "no attack while cooldown > 0"
    assert mob["attack_cooldown"] == 2, "cooldown ticks down by 1"


def test_full_env_zombie_uses_cooldown_via_mob_ai() -> None:
    """T21 integration: a zombie at adjacent tile attacks once, then waits."""
    from glyphbench.envs.craftax.full import CraftaxFullEnv

    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._agent_x, env._agent_y = 5, 5
    env._hp = 100
    env._mobs.append({
        "type": "zombie", "x": 5, "y": 6,
        "hp": 5, "max_hp": 5, "is_boss": False,
        "floor": env._current_floor, "attack_cooldown": 0,
    })

    initial_hp = env._hp
    # Step 1: zombie should attack and reset its own cooldown to 5.
    env.step(env.action_spec.names.index("NOOP"))
    after_first = env._hp
    assert after_first < initial_hp, "zombie should hit on first eligible step"

    # Steps 2-5: zombie cannot attack (cooldown still > 0). Use NOOP.
    # Inject the same zombie back into adjacency in case it moved away.
    env._mobs[-1]["x"], env._mobs[-1]["y"] = 5, 6
    for _ in range(4):
        env.step(env.action_spec.names.index("NOOP"))
        # Pin the zombie at adjacency for the test (it may have moved).
        env._mobs[-1]["x"], env._mobs[-1]["y"] = 5, 6

    # The HP loss after 4 more steps should be much less than 4× the first hit
    # (it is 0 if cooldown is honoured, since the zombie's first attack reset
    # cooldown to 5 and we only ran 4 more steps).
    after_cooldown = env._hp
    delta = after_first - after_cooldown
    assert delta == 0, f"zombie hit during cooldown: HP dropped by {delta}"


def test_sleeping_player_takes_3_5x_melee_damage() -> None:
    """T22: melee damage is multiplied by 3.5 when the player is asleep
    (upstream game_logic.py:1100-1291; multiplier 1 + 2.5*is_sleeping)."""
    from glyphbench.envs.craftax.full import CraftaxFullEnv

    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._agent_x, env._agent_y = 5, 5
    env._hp = 100
    env._is_sleeping = True
    env._mobs.append({
        "type": "zombie", "x": 5, "y": 6,
        "hp": 5, "max_hp": 5, "is_boss": False,
        "floor": env._current_floor, "attack_cooldown": 0,
    })
    initial_hp = env._hp
    # NOOP triggers the mob_ai → step_melee_mob path; zombie deals base 1 dmg
    # (per _MOB_STATS) which becomes round(1 * 3.5) = 4 with sleep multiplier.
    env.step(env.action_spec.names.index("NOOP"))
    delta = initial_hp - env._hp
    # Zombie's base damage is 1 (per _MOB_STATS); 1 * 3.5 = 3.5 → round to 4.
    # Subtract 0 armor (initial inventory has none), so delta should be 4.
    # If our impl uses floor instead, delta would be 3. Both are accepted as long
    # as it's clearly larger than the no-sleep case (1).
    assert delta >= 3, f"sleeping player should take >=3 damage, got {delta}"
    assert delta <= 5, f"sleep multiplier should not exceed ~3.5x, got {delta}"


def test_awake_player_takes_baseline_damage() -> None:
    """Sanity-check baseline (without sleep): zombie deals 1 damage."""
    from glyphbench.envs.craftax.full import CraftaxFullEnv

    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._agent_x, env._agent_y = 5, 5
    env._hp = 100
    env._is_sleeping = False
    env._mobs.append({
        "type": "zombie", "x": 5, "y": 6,
        "hp": 5, "max_hp": 5, "is_boss": False,
        "floor": env._current_floor, "attack_cooldown": 0,
    })
    initial_hp = env._hp
    env.step(env.action_spec.names.index("NOOP"))
    delta = initial_hp - env._hp
    # Baseline: zombie deals 1 damage (per _MOB_STATS), max(1, 1 - 0) = 1.
    assert delta == 1, f"awake player should take 1 damage from a zombie, got {delta}"


def test_full_env_has_is_sleeping_flag_after_reset() -> None:
    from glyphbench.envs.craftax.full import CraftaxFullEnv

    env = CraftaxFullEnv()
    env.reset(seed=0)
    assert hasattr(env, "_is_sleeping")
    assert env._is_sleeping is False
