"""Cross-benchmark normalization for paper-ready scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Reference:
    """Reference scores for normalization."""

    env_id: str
    random_return: float
    expert_return: float | None  # None = not yet sourced
    source: str  # citation or "placeholder"


# Placeholder reference table. Values should be updated from published papers.
REFERENCES: dict[str, Reference] = {
    # Atari (from Mnih et al. 2015 DQN paper where available)
    "atlas_rl/atari-pong-v0": Reference(
        "atlas_rl/atari-pong-v0", -20.7, 14.6, "Mnih et al. 2015"
    ),
    "atlas_rl/atari-breakout-v0": Reference(
        "atlas_rl/atari-breakout-v0", 1.7, 31.8, "Mnih et al. 2015"
    ),
    "atlas_rl/atari-spaceinvaders-v0": Reference(
        "atlas_rl/atari-spaceinvaders-v0", 148.0, 1652.0, "Mnih et al. 2015"
    ),
    "atlas_rl/atari-mspacman-v0": Reference(
        "atlas_rl/atari-mspacman-v0", 307.3, 15693.0, "Mnih et al. 2015"
    ),
    "atlas_rl/atari-qbert-v0": Reference(
        "atlas_rl/atari-qbert-v0", 163.9, 13455.0, "Mnih et al. 2015"
    ),
    "atlas_rl/atari-freeway-v0": Reference(
        "atlas_rl/atari-freeway-v0", 0.0, 29.6, "Mnih et al. 2015"
    ),
    # MiniGrid (random agent gets ~0, optimal gets ~1)
    "atlas_rl/minigrid-empty-5x5-v0": Reference(
        "atlas_rl/minigrid-empty-5x5-v0", 0.0, 0.95, "placeholder"
    ),
    "atlas_rl/minigrid-doorkey-5x5-v0": Reference(
        "atlas_rl/minigrid-doorkey-5x5-v0", 0.0, 0.90, "placeholder"
    ),
    "atlas_rl/minigrid-fourrooms-v0": Reference(
        "atlas_rl/minigrid-fourrooms-v0", 0.0, 0.85, "placeholder"
    ),
    # MiniHack (binary success: random ~0, expert 1)
    "atlas_rl/minihack-room-5x5-v0": Reference(
        "atlas_rl/minihack-room-5x5-v0", 0.0, 1.0, "placeholder"
    ),
    "atlas_rl/minihack-corridor-r2-v0": Reference(
        "atlas_rl/minihack-corridor-r2-v0", 0.0, 1.0, "placeholder"
    ),
    # Procgen (0-10 range typically)
    "atlas_rl/procgen-coinrun-v0": Reference(
        "atlas_rl/procgen-coinrun-v0", 0.0, 10.0, "placeholder"
    ),
    "atlas_rl/procgen-maze-v0": Reference(
        "atlas_rl/procgen-maze-v0", 0.0, 10.0, "placeholder"
    ),
    # Craftax (0-22 achievements)
    "atlas_rl/craftax-classic-v0": Reference(
        "atlas_rl/craftax-classic-v0", 0.0, 22.0, "placeholder"
    ),
}


def normalized_score(env_id: str, mean_return: float) -> float | None:
    """Compute normalized score for a single env.

    Returns None if no reference exists for this env.
    Score can be >1.0 (beats expert) or <0.0 (worse than random).
    """
    ref = REFERENCES.get(env_id)
    if ref is None or ref.expert_return is None:
        return None
    denom = ref.expert_return - ref.random_return
    if denom == 0.0:
        return None
    return (mean_return - ref.random_return) / denom


def benchmark_aggregate(
    scores: dict[str, float | None],
    method: str = "median",
) -> float:
    """Aggregate normalized scores across envs.

    Supports: median, mean, iqm (interquartile mean).
    Ignores None values.
    """
    valid = [s for s in scores.values() if s is not None]
    if not valid:
        return 0.0
    if method == "median":
        return float(np.median(valid))
    elif method == "iqm":
        sorted_s = sorted(valid)
        q = max(1, len(sorted_s) // 4)
        return float(np.mean(sorted_s[q : len(sorted_s) - q]))
    else:  # mean
        return float(np.mean(valid))
