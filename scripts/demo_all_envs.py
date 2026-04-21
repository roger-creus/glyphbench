#!/usr/bin/env python
"""Run 100 random steps on each env with Unicode + color terminal rendering.

Usage:
    uv run python scripts/demo_all_envs.py
    uv run python scripts/demo_all_envs.py --suite minigrid
    uv run python scripts/demo_all_envs.py --env glyphbench/atari-pong-v0
    uv run python scripts/demo_all_envs.py --pause  # wait for Enter between envs
"""

from __future__ import annotations

import argparse
import sys
import time

import glyphbench  # noqa: F401
import gymnasium as gym

from glyphbench.core import all_glyphbench_env_ids

# Import the color renderer from the replay utility
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from replay_trajectory import render_colored  # noqa: E402

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
# Hide cursor + move to top-left + erase from cursor to end of screen.
# Writing the full frame in one stdout.write after "\033[H" avoids the flash
# that `os.system("clear")` causes (spawns subprocess, blanks screen first).
CURSOR_HOME = "\033[H"
ERASE_TO_END = "\033[J"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"


def _build_frame(env_id: str, step: int, total_steps: int,
                 total_reward: float, action_names, obs: str,
                 status: str | None = None) -> str:
    header = (
        f"{ANSI_BOLD}{'=' * 70}{ANSI_RESET}\n"
        f"{ANSI_BOLD}{env_id}{ANSI_RESET}  "
        f"Step {step}/{total_steps}  "
    )
    if status:
        header += f"{status}{ANSI_RESET}  "
    header += f"Return: \033[1;33m{total_reward:+.2f}{ANSI_RESET}\n"
    header += f"Actions: \033[36m{action_names}{ANSI_RESET}\n"
    header += f"{ANSI_BOLD}{'=' * 70}{ANSI_RESET}\n\n"
    return header + render_colored(obs) + "\n"


def _draw(frame: str) -> None:
    # Single write: cursor home, paint frame, erase anything left over from a
    # taller previous frame. No intermediate blank state -> no flicker.
    sys.stdout.write(CURSOR_HOME + frame + ERASE_TO_END)
    sys.stdout.flush()


def run_env(env_id: str, steps: int = 100, delay: float = 0.1) -> float:
    env = gym.make(env_id, max_turns=steps)
    obs, info = env.reset(seed=42)

    action_names = env.unwrapped.action_spec.names
    total_reward = 0.0

    # Clear the screen once before the first frame so there's no leftover text.
    sys.stdout.write("\033[2J")
    _draw(_build_frame(env_id, 1, steps, total_reward, action_names, obs))

    for step in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            status = "\033[1;32mTERMINATED" if terminated else "\033[1;33mTRUNCATED"
            _draw(_build_frame(env_id, step + 1, steps, total_reward,
                               action_names, obs, status=status))
            break

        time.sleep(delay)
        _draw(_build_frame(env_id, step + 1, steps, total_reward,
                           action_names, obs))

    env.close()
    return total_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo all GlyphBench envs with random agent")
    parser.add_argument("--suite", type=str, help="Filter by suite (minigrid, atari, etc.)")
    parser.add_argument("--env", type=str, help="Run a single env")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per env")
    parser.add_argument("--delay", type=float, default=0.05, help="Seconds between steps")
    parser.add_argument("--pause", action="store_true", help="Pause between envs")
    args = parser.parse_args()

    if args.env:
        env_ids = [args.env]
    else:
        env_ids = all_glyphbench_env_ids()
        env_ids = [e for e in env_ids if "dummy" not in e]
        if args.suite:
            env_ids = [e for e in env_ids if args.suite in e]

    print(f"Running {len(env_ids)} environments, {args.steps} steps each\n")

    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()
    try:
        for i, env_id in enumerate(env_ids):
            if args.pause and i > 0:
                sys.stdout.write(SHOW_CURSOR)
                sys.stdout.flush()
                input(f"\n[{i}/{len(env_ids)}] Press Enter for next env...")
                sys.stdout.write(HIDE_CURSOR)
                sys.stdout.flush()

            try:
                ret = run_env(env_id, steps=args.steps, delay=args.delay)
                print(f"\n  Result: return={ret:+.3f}")
            except Exception as e:
                print(f"\n  ERROR on {env_id}: {e}")

            if not args.pause:
                time.sleep(0.5)
    finally:
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()

    print("\nDone!")


if __name__ == "__main__":
    main()
