#!/usr/bin/env python3
"""Play any glyphbench env with random actions, printing each step to the terminal.

Usage:
    uv run python scripts/play_random.py glyphbench/craftax-classic-v0
    uv run python scripts/play_random.py glyphbench/minigrid-doorkey-5x5-v0 --seed 42 --steps 50
    uv run python scripts/play_random.py glyphbench/minihack-eat-v0 --delay 0.3
"""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np

import glyphbench  # noqa: F401 — trigger env registration


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-agent viewer for glyphbench envs")
    parser.add_argument("env_id", help="Gym env ID, e.g. glyphbench/craftax-classic-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200, help="Max steps to run")
    parser.add_argument("--max-turns", type=int, default=500, help="Env max_turns")
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between steps (0 = no delay)")
    args = parser.parse_args()

    env = gym.make(args.env_id, max_turns=args.max_turns)
    unwrapped = env.unwrapped
    action_names = unwrapped.action_spec.names
    n_actions = unwrapped.action_spec.n
    rng = np.random.default_rng(args.seed)

    obs, info = env.reset(seed=args.seed)

    print(f"\n{'=' * 60}")
    print(f"  {args.env_id}  |  seed={args.seed}  |  actions={n_actions}")
    print(f"{'=' * 60}")
    print(f"\n--- SYSTEM PROMPT ---\n{unwrapped.system_prompt()[:400]}...")
    print(f"\n--- STEP 0 (reset) ---\n{obs}\n")

    total_reward = 0.0
    for step in range(1, args.steps + 1):
        action = int(rng.integers(0, n_actions))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"--- STEP {step}: {action_names[action]} | reward={reward:+.2f} | total={total_reward:.2f} ---")
        print(obs)

        if reward != 0:
            print(f"  >>> REWARD: {reward:+.2f}")
        if terminated:
            print(f"\n  TERMINATED (total reward: {total_reward:.2f})")
            break
        if truncated:
            print(f"\n  TRUNCATED at max turns (total reward: {total_reward:.2f})")
            break

        print()
        if args.delay > 0:
            time.sleep(args.delay)

    if not terminated and not truncated:
        print(f"\n  Stopped after {args.steps} steps (total reward: {total_reward:.2f})")


if __name__ == "__main__":
    main()
