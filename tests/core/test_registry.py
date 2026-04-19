
from atlas_rl.core.registry import (
    all_atlas_rl_env_ids,
    register_env,
)


def test_register_env_makes_id_visible_to_gym():
    register_env(
        "atlas_rl/__registry-test-1-v0",
        "tests.core.test_registry:_FakeEntry",
    )
    assert "atlas_rl/__registry-test-1-v0" in all_atlas_rl_env_ids()


def test_register_env_is_idempotent():
    register_env(
        "atlas_rl/__registry-test-2-v0",
        "tests.core.test_registry:_FakeEntry",
    )
    register_env(
        "atlas_rl/__registry-test-2-v0",
        "tests.core.test_registry:_FakeEntry",
    )
    ids = all_atlas_rl_env_ids()
    assert ids.count("atlas_rl/__registry-test-2-v0") == 1


def test_all_ids_are_sorted():
    register_env("atlas_rl/__zz-v0", "tests.core.test_registry:_FakeEntry")
    register_env("atlas_rl/__aa-v0", "tests.core.test_registry:_FakeEntry")
    ids = all_atlas_rl_env_ids()
    assert ids == sorted(ids)


class _FakeEntry:
    """Fake entry point used only so gym.register has something valid to point at."""
    pass
