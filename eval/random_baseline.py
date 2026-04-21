#!/usr/bin/env python
"""Run random baseline on all environments. Fast (no LLM needed).

Produces random_baseline.json: {env_id: {mean_return, std_return, ...}}
Used as the normalization anchor for all LLM eval scores.

Usage:
    uv run python eval/random_baseline.py
    uv run python eval/random_baseline.py --episodes 50 --suites minigrid atari
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import glyphbench  # noqa: F401 — register envs
from glyphbench.core import all_glyphbench_env_ids


def run_random_episodes(
    env_id: str, seeds: list[int], max_turns: int | None = None,
) -> dict[str, Any]:
    """Run random agent on one env across all seeds.

    If max_turns is None, each env uses its own natural budget (defined by
    the env class). Overriding it distorts the baseline for envs whose
    difficulty depends on the step budget.
    """
    returns = []
    lengths = []
    for seed in seeds:
        env = gym.make(env_id) if max_turns is None else gym.make(env_id, max_turns=max_turns)
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        returns.append(total_reward)
        lengths.append(steps)
        env.close()

    return {
        "env_id": env_id,
        "n_episodes": len(seeds),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "median_return": float(np.median(returns)),
        "mean_length": float(np.mean(lengths)),
        "per_episode_returns": [float(r) for r in returns],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GlyphBench random baseline")
    parser.add_argument("--episodes", type=int, default=25, help="Episodes per env")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Override env's natural max_turns (default: None = use env's own budget)")
    parser.add_argument("--suites", nargs="*", help="Filter by suite")
    parser.add_argument("--output", type=Path, default=Path("eval/random_baseline.json"))
    args = parser.parse_args()

    env_ids = all_glyphbench_env_ids()
    env_ids = [e for e in env_ids if "dummy" not in e]
    if args.suites:
        env_ids = [e for e in env_ids if any(s in e for s in args.suites)]

    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 2**31, size=args.episodes).tolist()

    print(f"Running random baseline on {len(env_ids)} envs, {args.episodes} episodes each")

    results: dict[str, Any] = {}
    start = time.time()

    for idx, env_id in enumerate(env_ids):
        t0 = time.time()
        try:
            res = run_random_episodes(env_id, seeds, args.max_turns)
            results[env_id] = res
            elapsed = time.time() - t0
            print(
                f"[{idx+1}/{len(env_ids)}] {env_id}: "
                f"return={res['mean_return']:+.3f} len={res['mean_length']:.0f} ({elapsed:.1f}s)"
            )
        except Exception as e:
            print(f"[{idx+1}/{len(env_ids)}] {env_id}: ERROR {e}")
            results[env_id] = {"env_id": env_id, "error": str(e)}

    total = time.time() - start
    print(f"\nDone in {total:.0f}s. {len(results)} envs.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
