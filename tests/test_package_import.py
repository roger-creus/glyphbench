def test_importing_glyphbench_registers_dummy_env():
    import glyphbench  # noqa: F401
    from glyphbench.core.registry import all_glyphbench_env_ids
    assert "glyphbench/__dummy-v0" in all_glyphbench_env_ids()


def test_make_env_can_make_dummy_env_via_id():
    from glyphbench.core import make_env

    import glyphbench  # noqa: F401
    env = make_env("glyphbench/__dummy-v0")
    obs, info = env.reset(0)
    assert isinstance(obs, str)
    assert "@" in obs


def test_core_public_reexports():
    from glyphbench.core import (
        ActionSpec,
        BaseGlyphEnv,
        GridObservation,
        all_glyphbench_env_ids,
        register_env,
    )
    # Sanity: each is what we expect
    assert callable(register_env)
    assert callable(all_glyphbench_env_ids)
    assert isinstance(GridObservation.__name__, str)
    assert isinstance(ActionSpec.__name__, str)
    assert isinstance(BaseGlyphEnv.__name__, str)
