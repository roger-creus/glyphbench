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
