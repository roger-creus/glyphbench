
from rl_world_ascii.core.registry import (
    all_rl_world_ascii_env_ids,
    register_env,
)


def test_register_env_makes_id_visible_to_gym():
    register_env(
        "rl_world_ascii/__registry-test-1-v0",
        "tests.core.test_registry:_FakeEntry",
    )
    assert "rl_world_ascii/__registry-test-1-v0" in all_rl_world_ascii_env_ids()


def test_register_env_is_idempotent():
    register_env(
        "rl_world_ascii/__registry-test-2-v0",
        "tests.core.test_registry:_FakeEntry",
    )
    register_env(
        "rl_world_ascii/__registry-test-2-v0",
        "tests.core.test_registry:_FakeEntry",
    )
    ids = all_rl_world_ascii_env_ids()
    assert ids.count("rl_world_ascii/__registry-test-2-v0") == 1


def test_all_ids_are_sorted():
    register_env("rl_world_ascii/__zz-v0", "tests.core.test_registry:_FakeEntry")
    register_env("rl_world_ascii/__aa-v0", "tests.core.test_registry:_FakeEntry")
    ids = all_rl_world_ascii_env_ids()
    assert ids == sorted(ids)


class _FakeEntry:
    """Fake entry point used only so gym.register has something valid to point at."""
    pass
