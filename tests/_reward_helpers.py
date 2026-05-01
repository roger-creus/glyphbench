"""Shared helpers for reward-bound auditing.

Lives outside ``test_*`` so non-test scripts (e.g. scripts/audit_reward_bounds.py)
can import the rollout helper without picking up pytest collection.
"""
from __future__ import annotations

import random

from glyphbench.core import make_env


def _random_rollout_return(env_id: str, seed: int = 0) -> tuple[float, int]:
    """Run a single random-action rollout; return (cumulative_reward, length)."""
    env = make_env(env_id)
    rng = random.Random(seed)
    env.reset(seed=seed)
    n_actions = env.action_spec.n
    total = 0.0
    length = 0
    for _ in range(env.max_turns):
        action = rng.randrange(n_actions)
        _, r, terminated, truncated, _ = env.step(action)
        total += float(r)
        length += 1
        if terminated or truncated:
            break
    return total, length
