def test_importing_rl_world_ascii_registers_dummy_env():
    import rl_world_ascii  # noqa: F401
    from rl_world_ascii.core.registry import all_rl_world_ascii_env_ids
    assert "rl_world_ascii/__dummy-v0" in all_rl_world_ascii_env_ids()


def test_gym_can_make_dummy_env_via_id():
    import gymnasium as gym

    import rl_world_ascii  # noqa: F401
    env = gym.make("rl_world_ascii/__dummy-v0")
    obs, info = env.reset(seed=0)
    assert isinstance(obs, str)
    assert "@" in obs


def test_core_public_reexports():
    from rl_world_ascii.core import (
        ActionSpec,
        BaseAsciiEnv,
        GridObservation,
        all_rl_world_ascii_env_ids,
        register_env,
    )
    # Sanity: each is what we expect
    assert callable(register_env)
    assert callable(all_rl_world_ascii_env_ids)
    assert isinstance(GridObservation.__name__, str)
    assert isinstance(ActionSpec.__name__, str)
    assert isinstance(BaseAsciiEnv.__name__, str)
