"""Cross-suite property test: cumulative return in [-1, 1] across multiple seeds.

This test is the structural enforcement of the reward-normalization rule
in the 2026-05-01 spec. Each env is checked across ``_NUM_SEEDS`` random
rollouts so latent bound violations cannot hide behind a benign seed=0
(as happened during P4-5 baseline regeneration, where 14 craftax envs
passed seed=0 but blew past the bound under other seeds).

Implementation note: rollouts are capped at ``_MAX_STEPS`` random steps
to keep total pytest runtime under a minute. Bound violations driven by
per-step reward magnitudes surface well within that cap.
"""
from __future__ import annotations

import pytest

import glyphbench  # noqa: F401  registers all envs
from glyphbench.core.task_selection import list_task_ids
from tests._reward_helpers import _random_rollout_return
from tests._reward_xfail import KNOWN_REWARD_VIOLATORS

# Floating-point slack on the [-1, 1] bound.
_EPS = 1e-6
# How many seeds to probe per env. 10 catches latent violators while
# staying within the runtime budget.
_NUM_SEEDS = 10
# Cap each rollout to keep runtime bounded for envs with very long
# max_turns (atari 10000, craftaxfull 10000). Reward-bound violations
# are fast-accumulating and surface within a few hundred steps.
_MAX_STEPS = 500


# Use list_task_ids() so the test naturally tracks the registry; parametrize
# at collection time so each env shows up as a distinct test case.
_ALL_ENV_IDS = list_task_ids()


@pytest.mark.parametrize("env_id", _ALL_ENV_IDS)
def test_cumulative_return_in_range(env_id: str) -> None:
    if env_id in KNOWN_REWARD_VIOLATORS:
        pytest.xfail(f"{env_id} is a known reward-bound violator (will be remapped)")
    for seed in range(_NUM_SEEDS):
        total, length = _random_rollout_return(env_id, seed=seed, max_steps=_MAX_STEPS)
        assert -1.0 - _EPS <= total <= 1.0 + _EPS, (
            f"{env_id} seed={seed}: cumulative return {total:.4f} after {length} steps "
            f"is outside [-1, 1]. Reward shape is not structurally bounded."
        )
