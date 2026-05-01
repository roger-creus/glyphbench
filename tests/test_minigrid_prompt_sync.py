"""Each minigrid env must include a [Legend] block in the per-turn obs,
not in the system prompt."""

from __future__ import annotations

from glyphbench.core.registry import make_env, all_glyphbench_env_ids
from glyphbench.envs import _import_all_suites

_import_all_suites()


def _minigrid_env_ids():
    return [eid for eid in all_glyphbench_env_ids() if eid.startswith("glyphbench/minigrid-")]


def test_system_prompt_does_not_embed_legend_for_minigrid():
    for env_id in _minigrid_env_ids():
        env = make_env(env_id)
        sp = env.system_prompt()
        # No glyph→meaning enumerations.
        assert " = " not in sp or not any(line.strip().startswith(("•", "-", "*")) for line in sp.splitlines() if " = " in line)


def test_per_turn_obs_has_legend_for_minigrid():
    for env_id in _minigrid_env_ids():
        env = make_env(env_id)
        obs_text, _ = env.reset(seed=0)
        assert "[Legend]" in obs_text, f"{env_id}: per-turn obs missing legend"
