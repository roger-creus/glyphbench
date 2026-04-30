"""Phase-α projectile entity + container tests."""
from glyphbench.envs.craftax.mechanics.projectiles import (
    ProjectileEntity,
    ProjectileType,
)


def test_projectile_entity_construction() -> None:
    p = ProjectileEntity(
        kind=ProjectileType.ARROW,
        x=5,
        y=7,
        dx=1,
        dy=0,
        damage=2,
    )
    assert p.kind == ProjectileType.ARROW
    assert (p.x, p.y) == (5, 7)
    assert (p.dx, p.dy) == (1, 0)
    assert p.damage == 2


def test_projectile_type_enum_has_eight_upstream_variants() -> None:
    """Upstream Craftax constants.py:125-134 defines 8 ProjectileType values."""
    expected = {"ARROW", "DAGGER", "FIREBALL", "ICEBALL",
               "ARROW2", "SLIMEBALL", "FIREBALL2", "ICEBALL2"}
    assert {p.name for p in ProjectileType} == expected


def test_projectile_advances_one_tile_per_step() -> None:
    p = ProjectileEntity(
        kind=ProjectileType.ARROW, x=5, y=7, dx=1, dy=0, damage=2,
    )
    p.advance()
    assert (p.x, p.y) == (6, 7)
    p.advance()
    assert (p.x, p.y) == (7, 7)
