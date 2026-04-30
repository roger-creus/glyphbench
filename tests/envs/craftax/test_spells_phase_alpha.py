"""Phase-α spell semantics tests."""
from glyphbench.envs.craftax.full import CraftaxFullEnv, Mob


def test_mob_typeddict_no_frozen_turns_field() -> None:
    """Iceball does not freeze upstream — frozen_turns has no semantic role."""
    assert "frozen_turns" not in Mob.__annotations__
