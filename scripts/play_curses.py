#!/usr/bin/env python3
"""Curses-based random-agent viewer for any atlas_rl environment.

Renders the grid with colors, a side panel for HUD/legend, and a bottom bar
for messages and reward tracking.

Usage:
    uv run python scripts/play_curses.py atlas_rl/craftax-classic-v0
    uv run python scripts/play_curses.py atlas_rl/minigrid-doorkey-5x5-v0 --seed 42
    uv run python scripts/play_curses.py atlas_rl/atari-pong-v0 --delay 0.05
"""

from __future__ import annotations

import argparse
import curses
import time

import gymnasium as gym
import numpy as np

import atlas_rl  # noqa: F401 — trigger env registration
from terminal_colors import char_attr, init_colors


def _parse_obs(obs: str) -> dict[str, str]:
    """Parse a rendered observation into sections."""
    sections: dict[str, str] = {}
    current_key = ""
    current_lines: list[str] = []

    for line in obs.split("\n"):
        if line.startswith("[") and line.endswith("]"):
            if current_key:
                sections[current_key] = "\n".join(current_lines)
            current_key = line[1:-1]
            current_lines = []
        else:
            current_lines.append(line)

    if current_key:
        sections[current_key] = "\n".join(current_lines)
    return sections


def _draw_grid(win: curses.window, grid_str: str, y0: int, x0: int) -> None:
    """Draw the grid with per-character coloring."""
    max_y, max_x = win.getmaxyx()
    for row_idx, line in enumerate(grid_str.split("\n")):
        y = y0 + row_idx
        if y >= max_y - 1:
            break
        for col_idx, ch in enumerate(line):
            x = x0 + col_idx
            if x >= max_x - 1:
                break
            try:
                win.addch(y, x, ch, char_attr(ch))
            except curses.error:
                pass


def _draw_text(
    win: curses.window,
    text: str,
    y0: int,
    x0: int,
    attr: int = 0,
    max_width: int = 0,
) -> int:
    """Draw wrapped text, return number of lines used."""
    max_y, max_x = win.getmaxyx()
    if max_width <= 0:
        max_width = max_x - x0 - 1
    lines_used = 0
    for line in text.split("\n"):
        if not line:
            lines_used += 1
            continue
        # Word-wrap
        while line:
            chunk = line[:max_width]
            line = line[max_width:]
            y = y0 + lines_used
            if y >= max_y - 1:
                return lines_used
            try:
                win.addnstr(y, x0, chunk, max_width, attr)
            except curses.error:
                pass
            lines_used += 1
    return lines_used


def _draw_hbar(win: curses.window, y: int, x: int, width: int) -> None:
    """Draw a horizontal bar."""
    max_y, max_x = win.getmaxyx()
    if y >= max_y - 1:
        return
    bar = "\u2500" * min(width, max_x - x - 1)
    try:
        win.addstr(y, x, bar, curses.A_DIM)
    except curses.error:
        pass


def main(stdscr: curses.window) -> None:
    parser = argparse.ArgumentParser(
        description="Curses random-agent viewer for atlas_rl envs"
    )
    parser.add_argument("env_id", help="Gym env ID")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument(
        "--delay", type=float, default=0.1,
        help="Seconds between steps",
    )
    args = parser.parse_args()

    # Curses setup
    init_colors()
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(int(args.delay * 1000))

    # Env setup
    env = gym.make(args.env_id, max_turns=args.max_turns)
    unwrapped = env.unwrapped
    action_names = unwrapped.action_spec.names
    n_actions = unwrapped.action_spec.n
    rng = np.random.default_rng(args.seed)

    obs, info = env.reset(seed=args.seed)
    total_reward = 0.0
    step_num = 0
    last_action = "(reset)"
    last_reward = 0.0
    done = False

    while True:
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()
        if max_y < 10 or max_x < 40:
            stdscr.addstr(0, 0, "Terminal too small")
            stdscr.refresh()
            time.sleep(0.5)
            continue

        sections = _parse_obs(obs)
        grid_str = sections.get("Grid", "")
        grid_lines = grid_str.split("\n")
        grid_h = len(grid_lines)
        grid_w = max((len(l) for l in grid_lines), default=0)

        # Layout: grid on the left, info panel on the right
        panel_x = min(grid_w + 3, max_x * 2 // 3)
        panel_w = max_x - panel_x - 1

        # Title bar
        title = f" {args.env_id} | seed={args.seed} | step={step_num} "
        try:
            stdscr.addstr(
                0, 0, title.center(max_x - 1),
                curses.A_REVERSE | curses.A_BOLD,
            )
        except curses.error:
            pass

        # Grid
        _draw_grid(stdscr, grid_str, 2, 1)

        # Action + reward bar below grid
        bar_y = max(grid_h + 3, 2)
        _draw_hbar(stdscr, bar_y, 0, max_x)
        action_str = f" Action: {last_action}"
        reward_str = f"R={last_reward:+.2f}  Total={total_reward:.2f}"
        try:
            stdscr.addstr(
                bar_y + 1, 1, action_str,
                curses.color_pair(3) | curses.A_BOLD,
            )
            stdscr.addstr(
                bar_y + 1, max(len(action_str) + 3, max_x - len(reward_str) - 2),
                reward_str,
                curses.color_pair(2) | curses.A_BOLD if last_reward > 0
                else curses.color_pair(1) | curses.A_BOLD if last_reward < 0
                else curses.A_DIM,
            )
        except curses.error:
            pass

        # Message bar
        message = sections.get("Message", "").strip()
        if message:
            try:
                stdscr.addnstr(
                    bar_y + 2, 1, f" {message}",
                    max_x - 2, curses.color_pair(3),
                )
            except curses.error:
                pass

        # Right panel: HUD
        panel_y = 2
        hud = sections.get("HUD", "")
        if hud and panel_w > 10:
            try:
                stdscr.addstr(
                    panel_y, panel_x,
                    " HUD ",
                    curses.A_REVERSE,
                )
            except curses.error:
                pass
            panel_y += 1
            panel_y += _draw_text(
                stdscr, hud.strip(), panel_y, panel_x,
                max_width=panel_w,
            )
            panel_y += 1

        # Right panel: Legend
        legend = sections.get("Legend", "")
        if legend and panel_w > 10 and panel_y < max_y - 3:
            try:
                stdscr.addstr(
                    panel_y, panel_x,
                    " Legend ",
                    curses.A_REVERSE,
                )
            except curses.error:
                pass
            panel_y += 1
            for legend_line in legend.strip().split("\n"):
                if panel_y >= max_y - 2:
                    break
                # Color the symbol char in the legend
                if len(legend_line) >= 1:
                    sym = legend_line[0]
                    try:
                        stdscr.addch(panel_y, panel_x, sym, char_attr(sym))
                        stdscr.addnstr(
                            panel_y, panel_x + 1,
                            legend_line[1:min(len(legend_line), panel_w)],
                            panel_w - 1,
                        )
                    except curses.error:
                        pass
                panel_y += 1

        # Status bar at bottom
        if done:
            status = " DONE (press q to quit) "
            attr = curses.A_REVERSE | curses.color_pair(1)
        else:
            status = f" Actions: {n_actions} | Press q to quit, r to reset "
            attr = curses.A_REVERSE
        try:
            stdscr.addstr(max_y - 1, 0, status.ljust(max_x - 1), attr)
        except curses.error:
            pass

        stdscr.refresh()

        # Handle input
        key = stdscr.getch()
        if key == ord("q"):
            break
        if key == ord("r"):
            obs, info = env.reset(seed=args.seed)
            total_reward = 0.0
            step_num = 0
            last_action = "(reset)"
            last_reward = 0.0
            done = False
            continue

        if done:
            continue

        # Take random action
        action = int(rng.integers(0, n_actions))
        last_action = action_names[action]
        obs, reward, terminated, truncated, info = env.step(action)
        last_reward = reward
        total_reward += reward
        step_num += 1

        if terminated or truncated:
            done = True


if __name__ == "__main__":
    curses.wrapper(main)
