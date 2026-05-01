"""Cross-suite property test: every env keeps cumulative return in [-1, 1].

This test is the structural enforcement of the reward-normalization rule
in the 2026-05-01 spec. It will fail for envs that have not yet been
remapped — that is intentional. Each P2/P3/P4 task is gated by re-running
this test after the edit.

Implementation note: we run a single random rollout per env with a fixed
seed for reproducibility, and use a small tolerance to absorb floating
point error.
"""
from __future__ import annotations

import pytest

import glyphbench  # registers all envs
from glyphbench.core.task_selection import list_task_ids
from tests._reward_helpers import _random_rollout_return
from tests._reward_xfail import KNOWN_REWARD_VIOLATORS

# Floating-point slack on the [-1, 1] bound.
_EPS = 1e-6

# Use list_task_ids() so the test naturally tracks the registry; parametrize
# at collection time so each env shows up as a distinct test case.
_ALL_ENV_IDS = list_task_ids()


@pytest.mark.parametrize("env_id", _ALL_ENV_IDS)
def test_cumulative_return_in_range(env_id: str) -> None:
    if env_id in KNOWN_REWARD_VIOLATORS:
        pytest.xfail(f"{env_id} is a known reward-bound violator (will be remapped)")
    total, length = _random_rollout_return(env_id, seed=0)
    assert -1.0 - _EPS <= total <= 1.0 + _EPS, (
        f"{env_id}: cumulative return {total:.4f} after {length} steps "
        f"is outside [-1, 1]. Reward shape needs remapping."
    )
