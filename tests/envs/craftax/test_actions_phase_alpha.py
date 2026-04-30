"""Phase-α action-spec adjustments."""
from glyphbench.envs.craftax.base import CRAFTAX_FULL_ACTION_SPEC


def test_cast_heal_removed_from_full_action_spec() -> None:
    """Upstream has no heal spell — only potions."""
    assert "CAST_HEAL" not in CRAFTAX_FULL_ACTION_SPEC.names
