#!/usr/bin/env python3
"""Interactive curses-based player for any glyphbench environment.

Play any environment with keyboard controls. The action menu shows
numbered actions; press the corresponding number key to act.

Usage:
    uv run python scripts/play_interactive.py glyphbench/craftax-classic-v0
    uv run python scripts/play_interactive.py glyphbench/minigrid-doorkey-5x5-v0 --seed 42
"""

from __future__ import annotations

import argparse
import curses

import gymnasium as gym

import glyphbench  # noqa: F401 — trigger env registration
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


# Common keyboard shortcuts for movement actions
_KEY_TO_ACTION: dict[int, str] = {
    curses.KEY_UP: "MOVE_UP",
    curses.KEY_DOWN: "MOVE_DOWN",
    curses.KEY_LEFT: "MOVE_LEFT",
    curses.KEY_RIGHT: "MOVE_RIGHT",
    ord("w"): "MOVE_UP",
    ord("s"): "MOVE_DOWN",
    ord("a"): "MOVE_LEFT",
    ord("d"): "MOVE_RIGHT",
    ord("W"): "MOVE_UP",
    ord("S"): "MOVE_DOWN",
    ord("A"): "MOVE_LEFT",
    ord("D"): "MOVE_RIGHT",
    # MiniGrid: forward/left/right/toggle/pickup/drop
    ord("f"): "FORWARD",
    ord("F"): "FORWARD",
    ord("t"): "TOGGLE",
    ord("T"): "TOGGLE",
    ord("p"): "PICKUP",
    ord("P"): "PICKUP",
    ord("x"): "DROP",
    ord("X"): "DROP",
    # MiniGrid turn
    ord(","): "LEFT",
    ord("."): "RIGHT",
    # General
    ord(" "): "NOOP",
    ord("e"): "DO",
    ord("E"): "DO",
    # Minihack directional
    ord("h"): "WEST",
    ord("j"): "SOUTH",
    ord("k"): "NORTH",
    ord("l"): "EAST",
}


def main(stdscr: curses.window) -> None:
    parser = argparse.ArgumentParser(
        description="Interactive player for glyphbench envs"
    )
    parser.add_argument("env_id", help="Gym env ID")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=500)
    args = parser.parse_args()

    # Curses setup
    init_colors()
    curses.curs_set(0)
    stdscr.keypad(True)

    # Env setup
    env = gym.make(args.env_id, max_turns=args.max_turns)
    unwrapped = env.unwrapped
    action_names = unwrapped.action_spec.names
    n_actions = unwrapped.action_spec.n

    # Build name -> index lookup
    name_to_idx: dict[str, int] = {
        name: idx for idx, name in enumerate(action_names)
    }

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
            stdscr.getch()
            continue

        sections = _parse_obs(obs)
        grid_str = sections.get("Grid", "")
        grid_lines = grid_str.split("\n")
        grid_h = len(grid_lines)
        grid_w = max((len(l) for l in grid_lines), default=0)

        # Layout: grid left, info right
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

        # Right panel: HUD
        panel_y = 2
        hud = sections.get("HUD", "")
        if hud and panel_w > 10:
            try:
                stdscr.addstr(panel_y, panel_x, " HUD ", curses.A_REVERSE)
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
        if legend and panel_w > 10 and panel_y < max_y - 5:
            try:
                stdscr.addstr(panel_y, panel_x, " Legend ", curses.A_REVERSE)
            except curses.error:
                pass
            panel_y += 1
            for legend_line in legend.strip().split("\n"):
                if panel_y >= max_y - n_actions - 4:
                    break
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

        # Action + reward info below grid
        info_y = max(grid_h + 3, 2)
        action_line = f" Last: {last_action}"
        reward_line = f"R={last_reward:+.2f}  Total={total_reward:.2f}"
        try:
            stdscr.addstr(
                info_y, 1, action_line,
                curses.color_pair(3) | curses.A_BOLD,
            )
            rattr = (
                curses.color_pair(2) | curses.A_BOLD if last_reward > 0
                else curses.color_pair(1) | curses.A_BOLD if last_reward < 0
                else curses.A_DIM
            )
            r_x = max(len(action_line) + 3, max_x - len(reward_line) - 2)
            stdscr.addstr(info_y, r_x, reward_line, rattr)
        except curses.error:
            pass

        # Message
        message = sections.get("Message", "").strip()
        if message:
            try:
                stdscr.addnstr(
                    info_y + 1, 1, f" {message}",
                    max_x - 2, curses.color_pair(3),
                )
            except curses.error:
                pass

        # Action menu below grid/info
        menu_y = info_y + 3
        if menu_y < max_y - 2 and not done:
            try:
                stdscr.addstr(
                    menu_y, 1, " Actions ",
                    curses.A_REVERSE,
                )
            except curses.error:
                pass
            menu_y += 1
            for idx, name in enumerate(action_names):
                if menu_y >= max_y - 1:
                    break
                label = f" {idx}: {name}"
                try:
                    stdscr.addnstr(
                        menu_y, 1, label, max_x - 2,
                        curses.A_BOLD if idx < 10 else 0,
                    )
                except curses.error:
                    pass
                menu_y += 1

        # Status bar
        if done:
            status = " DONE | q=quit r=reset "
            attr = curses.A_REVERSE | curses.color_pair(1)
        else:
            status = (
                " arrows/wasd=move  0-9=action  "
                "space=noop  e=DO  q=quit  r=reset "
            )
            attr = curses.A_REVERSE
        try:
            stdscr.addstr(max_y - 1, 0, status.ljust(max_x - 1), attr)
        except curses.error:
            pass

        stdscr.refresh()

        # Wait for input
        key = stdscr.getch()
        if key == ord("q") or key == ord("Q"):
            break
        if key == ord("r") or key == ord("R"):
            obs, info = env.reset(seed=args.seed)
            total_reward = 0.0
            step_num = 0
            last_action = "(reset)"
            last_reward = 0.0
            done = False
            continue

        if done:
            continue

        # Resolve action
        action_idx: int | None = None

        # Check keyboard shortcuts first
        if key in _KEY_TO_ACTION:
            action_name = _KEY_TO_ACTION[key]
            if action_name in name_to_idx:
                action_idx = name_to_idx[action_name]

        # Number keys 0-9 for action by index
        if action_idx is None and ord("0") <= key <= ord("9"):
            idx = key - ord("0")
            if idx < n_actions:
                action_idx = idx

        if action_idx is None:
            continue

        last_action = action_names[action_idx]
        obs, reward, terminated, truncated, info = env.step(action_idx)
        last_reward = reward
        total_reward += reward
        step_num += 1

        if terminated or truncated:
            done = True


if __name__ == "__main__":
    curses.wrapper(main)
