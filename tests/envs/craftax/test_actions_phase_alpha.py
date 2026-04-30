"""Phase-α action-spec adjustments."""
from glyphbench.envs.craftax.base import CRAFTAX_FULL_ACTION_SPEC
from glyphbench.envs.craftax.full import CraftaxFullEnv


def test_cast_heal_removed_from_full_action_spec() -> None:
    """Upstream has no heal spell — only potions."""
    assert "CAST_HEAL" not in CRAFTAX_FULL_ACTION_SPEC.names


def test_make_spell_scroll_removed_from_full_action_spec() -> None:
    """Upstream has no spell scroll — books from chests are the only spell-learn pipeline."""
    assert "MAKE_SPELL_SCROLL" not in CRAFTAX_FULL_ACTION_SPEC.names


def test_full_env_inventory_has_bow_arrows_torch_after_reset() -> None:
    """Phase α adds bow / arrows / torch as first-class inventory keys.
    The keys must exist (not just default to 0 via .get) so future code can
    iterate over inventory items without missing the new ones.
    """
    env = CraftaxFullEnv()
    env.reset(seed=0)
    assert "bow" in env._inventory and env._inventory["bow"] == 0
    assert "arrows" in env._inventory and env._inventory["arrows"] == 0
    assert "torch" in env._inventory and env._inventory["torch"] == 0
