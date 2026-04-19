def test_importing_atlas_rl_registers_dummy_env():
    import atlas_rl  # noqa: F401
    from atlas_rl.core.registry import all_atlas_rl_env_ids
    assert "atlas_rl/__dummy-v0" in all_atlas_rl_env_ids()


def test_gym_can_make_dummy_env_via_id():
    import gymnasium as gym

    import atlas_rl  # noqa: F401
    env = gym.make("atlas_rl/__dummy-v0")
    obs, info = env.reset(seed=0)
    assert isinstance(obs, str)
    assert "@" in obs


def test_core_public_reexports():
    from atlas_rl.core import (
        ActionSpec,
        BaseAsciiEnv,
        GridObservation,
        all_atlas_rl_env_ids,
        register_env,
    )
    # Sanity: each is what we expect
    assert callable(register_env)
    assert callable(all_atlas_rl_env_ids)
    assert isinstance(GridObservation.__name__, str)
    assert isinstance(ActionSpec.__name__, str)
    assert isinstance(BaseAsciiEnv.__name__, str)
