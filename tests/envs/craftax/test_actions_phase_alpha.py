"""Phase-α action-spec adjustments."""
from glyphbench.envs.craftax.base import CRAFTAX_FULL_ACTION_SPEC, TILE_TABLE
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


def test_make_arrow_consumes_wood_and_stone_grants_two_arrows() -> None:
    """Upstream MAKE_ARROW: 1 wood + 1 stone -> 2 arrows, requires adjacent table."""
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._inventory["wood"] = 2
    env._inventory["stone"] = 2
    # Place a crafting table adjacent to the agent.
    grid = env._current_grid()
    target_x, target_y = env._agent_x + 1, env._agent_y
    grid[target_y][target_x] = TILE_TABLE  # use the existing TILE_TABLE constant

    action_idx = env.action_spec.names.index("MAKE_ARROW")
    env.step(action_idx)

    assert env._inventory["wood"] == 1
    assert env._inventory["stone"] == 1
    assert env._inventory["arrows"] == 2


def test_make_arrow_no_op_without_adjacent_table() -> None:
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._inventory["wood"] = 2
    env._inventory["stone"] = 2
    # No table adjacent; reset() does not place one near agent.
    initial = dict(env._inventory)

    action_idx = env.action_spec.names.index("MAKE_ARROW")
    env.step(action_idx)

    # Nothing consumed, no arrows granted.
    assert env._inventory.get("arrows", 0) == 0
    assert env._inventory["wood"] == initial["wood"]
    assert env._inventory["stone"] == initial["stone"]


def test_make_arrow_no_op_without_materials() -> None:
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._inventory["wood"] = 0
    env._inventory["stone"] = 5
    grid = env._current_grid()
    grid[env._agent_y][env._agent_x + 1] = TILE_TABLE

    action_idx = env.action_spec.names.index("MAKE_ARROW")
    env.step(action_idx)

    assert env._inventory.get("arrows", 0) == 0
    assert env._inventory["stone"] == 5  # nothing consumed
