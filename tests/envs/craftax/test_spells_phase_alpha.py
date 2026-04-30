"""Phase-α spell semantics tests."""
from glyphbench.envs.craftax.base import CRAFTAX_FULL_ACTION_SPEC
from glyphbench.envs.craftax.full import CraftaxFullEnv, Mob


def test_mob_typeddict_no_frozen_turns_field() -> None:
    """Iceball does not freeze upstream — frozen_turns has no semantic role."""
    assert "frozen_turns" not in Mob.__annotations__


def test_cast_iceball_placeholder_is_a_no_op() -> None:
    """T06 leaves CAST_ICEBALL as a placeholder until T11 reintroduces it as a projectile.

    Call the handler directly (not via env.step) so the per-step achievement-message
    machinery does not overwrite the placeholder text. The handler-level contract is:
    return 0.0, do not consume mana, set a player-facing (non-dev) message.
    """
    env = CraftaxFullEnv()
    env.reset(seed=0)
    env._mana = 5
    initial_mana = env._mana

    reward = env._handle_cast_iceball()

    assert reward == 0.0, "placeholder must return 0 reward"
    assert env._mana == initial_mana, "placeholder must not consume mana"
    assert "iceball" in env._message.lower(), "message should mention iceball"
    # No internal-development metadata should leak into agent-visible text.
    assert "phase" not in env._message.lower()
    assert "placeholder" not in env._message.lower()
