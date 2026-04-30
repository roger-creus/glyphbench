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


def test_make_torch_consumes_wood_and_coal_grants_four_torches() -> None:
    """Upstream MAKE_TORCH: 1 wood + 1 coal -> 4 torches, requires adjacent table."""
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._inventory["wood"] = 2
    env._inventory["coal"] = 2
    grid = env._current_grid()
    grid[env._agent_y][env._agent_x + 1] = TILE_TABLE

    action_idx = env.action_spec.names.index("MAKE_TORCH")
    env.step(action_idx)

    assert env._inventory["wood"] == 1
    assert env._inventory["coal"] == 1
    assert env._inventory["torch"] == 4


def test_make_torch_no_op_without_adjacent_table() -> None:
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._inventory["wood"] = 2
    env._inventory["coal"] = 2
    initial = dict(env._inventory)

    action_idx = env.action_spec.names.index("MAKE_TORCH")
    env.step(action_idx)

    assert env._inventory.get("torch", 0) == 0
    assert env._inventory["wood"] == initial["wood"]
    assert env._inventory["coal"] == initial["coal"]


def test_make_torch_no_op_without_materials() -> None:
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._inventory["wood"] = 5
    env._inventory["coal"] = 0
    grid = env._current_grid()
    grid[env._agent_y][env._agent_x + 1] = TILE_TABLE

    action_idx = env.action_spec.names.index("MAKE_TORCH")
    env.step(action_idx)

    assert env._inventory.get("torch", 0) == 0
    assert env._inventory["wood"] == 5


def test_place_torch_consumes_one_torch_not_raw_materials() -> None:
    """Upstream PLACE_TORCH consumes a crafted torch, not 1 wood + 1 coal."""
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._inventory["torch"] = 2
    env._inventory["wood"] = 0
    env._inventory["coal"] = 0
    # Position the agent on overworld (floor 0); the cell in front is grass by default.
    fx = env._agent_x + env._facing[0]
    fy = env._agent_y + env._facing[1]
    grid = env._current_grid()
    # Ensure the front tile is plain grass (placeable).
    from glyphbench.envs.craftax.base import TILE_GRASS
    grid[fy][fx] = TILE_GRASS

    action_idx = env.action_spec.names.index("PLACE_TORCH")
    env.step(action_idx)

    assert env._inventory["torch"] == 1, "one torch consumed"
    assert env._inventory["wood"] == 0
    assert env._inventory["coal"] == 0


def test_place_torch_no_op_without_torch_in_inventory() -> None:
    """Wood + coal alone are no longer enough — must be a crafted torch."""
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._inventory["torch"] = 0
    env._inventory["wood"] = 5
    env._inventory["coal"] = 5
    fx = env._agent_x + env._facing[0]
    fy = env._agent_y + env._facing[1]
    grid = env._current_grid()
    from glyphbench.envs.craftax.base import TILE_GRASS
    grid[fy][fx] = TILE_GRASS

    action_idx = env.action_spec.names.index("PLACE_TORCH")
    env.step(action_idx)

    assert env._inventory["wood"] == 5, "wood NOT consumed without torch"
    assert env._inventory["coal"] == 5, "coal NOT consumed without torch"
    assert env._inventory["torch"] == 0


def test_place_torch_no_op_when_target_tile_not_empty() -> None:
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._inventory["torch"] = 1
    fx = env._agent_x + env._facing[0]
    fy = env._agent_y + env._facing[1]
    grid = env._current_grid()
    from glyphbench.envs.craftax.base import TILE_STONE
    grid[fy][fx] = TILE_STONE  # solid, not placeable

    action_idx = env.action_spec.names.index("PLACE_TORCH")
    env.step(action_idx)

    assert env._inventory["torch"] == 1, "torch NOT consumed when placement fails"


def test_shoot_arrow_spawns_arrow_projectile_consumes_one_arrow() -> None:
    """Upstream SHOOT_ARROW: requires bow + 1 arrow, spawns ARROW projectile at player tile."""
    from glyphbench.envs.craftax.mechanics.projectiles import ProjectileType

    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._inventory["bow"] = 1
    env._inventory["arrows"] = 3
    env._facing = (1, 0)
    env._agent_x, env._agent_y = 5, 5

    reward = env._handle_shoot_arrow()

    assert isinstance(reward, float)
    assert env._inventory["arrows"] == 2
    assert len(env._player_projectiles) == 1
    p = env._player_projectiles[0]
    assert p.kind == ProjectileType.ARROW
    # Upstream-faithful spawn: at the player's tile, NOT player+dir.
    # The per-step driver will advance it +dir on the same step.
    assert (p.x, p.y) == (5, 5)
    assert (p.dx, p.dy) == (1, 0)


def test_shoot_arrow_no_op_without_bow() -> None:
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._inventory["bow"] = 0
    env._inventory["arrows"] = 5
    initial_arrows = env._inventory["arrows"]

    env._handle_shoot_arrow()

    assert env._inventory["arrows"] == initial_arrows, "no arrows consumed without bow"
    assert env._player_projectiles == []


def test_shoot_arrow_no_op_without_arrows() -> None:
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._inventory["bow"] = 1
    env._inventory["arrows"] = 0

    env._handle_shoot_arrow()

    assert env._player_projectiles == []


def test_full_action_spec_size_and_ordering_post_phase_alpha() -> None:
    """Phase α net: -2 (CAST_HEAL T04, MAKE_SPELL_SCROLL T05) + 3
    (MAKE_ARROW T14, MAKE_TORCH T15, SHOOT_ARROW T17) = +1.
    Original full spec was 35; phase α leaves it at 36.
    Phase β T02β adds REST (+1) → 37. Phase β T08β drops DRINK_POTION and
    adds 6 DRINK_POTION_* color actions (net +5) → 42.
    Phase β T11β adds READ_BOOK (+1) → 43.
    Phase γ will add LEVEL_UP_* (3) + ENCHANT_BOW.
    """
    spec = CRAFTAX_FULL_ACTION_SPEC
    assert len(spec.names) == 43, f"expected 43, got {len(spec.names)}: {spec.names}"
    # Required new names present (T14/T15/T17):
    for name in ("SHOOT_ARROW", "MAKE_ARROW", "MAKE_TORCH"):
        assert name in spec.names, f"missing required action {name!r}"
    # Spell + place actions retained (semantics changed but names same):
    for name in ("CAST_FIREBALL", "CAST_ICEBALL", "PLACE_TORCH"):
        assert name in spec.names, f"missing required action {name!r}"
    # Removed names absent (T04/T05):
    for name in ("CAST_HEAL", "MAKE_SPELL_SCROLL"):
        assert name not in spec.names, f"removed action {name!r} still in spec"
    # Names tuple length matches descriptions tuple length.
    assert len(spec.names) == len(spec.descriptions), (
        f"names ({len(spec.names)}) and descriptions ({len(spec.descriptions)}) misaligned"
    )


def test_full_action_spec_descriptions_well_formed() -> None:
    """Each description is a non-empty string."""
    spec = CRAFTAX_FULL_ACTION_SPEC
    for name, desc in zip(spec.names, spec.descriptions):
        assert isinstance(desc, str), f"description for {name} is not a str"
        assert desc.strip(), f"description for {name} is empty/whitespace"


def test_full_env_prompt_mentions_new_phase_alpha_actions() -> None:
    """T27: phase-α actions are present, removed actions are absent."""
    env = CraftaxFullEnv()
    env.reset(seed=0)
    # The system prompt is built by the env. Different envs surface this
    # differently — try common attribute names then fall back to a public
    # method.
    if hasattr(env, "system_prompt"):
        raw = env.system_prompt
        prompt = raw() if callable(raw) else raw
    elif hasattr(env, "_build_system_prompt"):
        prompt = env._build_system_prompt()
    elif hasattr(env, "system_prompt_template"):
        prompt = env.system_prompt_template
    else:
        # Last resort — check the static class attribute.
        prompt = getattr(env, "_SYSTEM_PROMPT", "") or getattr(env.__class__, "system_prompt", "")
    assert prompt, "could not find system prompt on env"

    # All phase-α actions must be mentioned.
    for token in (
        "SHOOT_ARROW", "MAKE_ARROW", "MAKE_TORCH",
        "PLACE_TORCH", "CAST_FIREBALL", "CAST_ICEBALL",
    ):
        assert token in prompt, f"prompt missing required token {token!r}"

    # Removed actions must NOT be mentioned.
    for stale in ("MAKE_SPELL_SCROLL", "CAST_HEAL"):
        assert stale not in prompt, f"prompt still mentions removed action {stale!r}"

    # Removed mechanics must NOT be advertised:
    # - "freeze" referring to iceball was removed in T06.
    # - AOE / radius for fireball was removed in T10.
    # Be loose: the words can appear (e.g. "freeze" might describe weather)
    # but they should NOT appear adjacent to "iceball" / "fireball".
    lower = prompt.lower()
    if "iceball" in lower:
        # Look for a window of ~80 chars around each "iceball" mention and
        # assert "freeze" is not in that window.
        idx = 0
        while True:
            idx = lower.find("iceball", idx)
            if idx < 0:
                break
            window = lower[max(0, idx - 40):idx + 40]
            assert "freeze" not in window, (
                f"prompt advertises freeze near iceball: ...{window}..."
            )
            idx += 1
    if "fireball" in lower:
        idx = 0
        while True:
            idx = lower.find("fireball", idx)
            if idx < 0:
                break
            window = lower[max(0, idx - 40):idx + 40]
            for stale_phrase in ("aoe", "radius", "2-tile", "area damage"):
                assert stale_phrase not in window, (
                    f"prompt advertises {stale_phrase!r} near fireball: ...{window}..."
                )
            idx += 1
