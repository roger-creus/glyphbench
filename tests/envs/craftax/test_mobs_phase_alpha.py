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
