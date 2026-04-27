"""GlyphBench CLI: list/replay helpers + submission bundling.

Eval execution is delegated to `vf-eval glyphbench` (verifiers' standard
runner) — see eval/run_debug.sh / eval/run_full.sh for the canonical
invocations. The CLI here covers introspection + visualisation only.

Examples:
    glyphbench list-suites
    glyphbench list-envs --suite atari
    glyphbench replay cluster_manager/results --suite minigrid --pause
    glyphbench bundle results/<model>/<run-hash>
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ALL_SUITES = ["minigrid", "minihack", "atari", "procgen", "craftax", "classics"]


def _slug_model(model: str) -> str:
    return model.replace("/", "_")


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------


def _git_commit_sha(repo_root: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _glyphbench_version() -> str | None:
    try:
        return metadata.version("glyphbench")
    except metadata.PackageNotFoundError:
        return None


def _bundle_dir(results_dir: Path, tar_output: Path | None) -> Path:
    """Package one results dir into a submission tarball.

    Expects results_dir to contain results.json + per_env/ + trajectories/
    (the layout produced by `prime eval run` / `vf-eval`).
    """
    if not (results_dir / "results.json").exists():
        raise FileNotFoundError(
            f"{results_dir}/results.json not found; pass the per-harness dir"
        )

    agg = json.loads((results_dir / "results.json").read_text())
    model = agg.get("model", "unknown")
    harness = results_dir.name
    model_slug = _slug_model(model)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    meta = {
        "model": model,
        "harness": harness,
        "episodes_per_env": agg.get("episodes_per_env"),
        "n_envs": agg.get("n_envs"),
        "temperature": agg.get("temperature"),
        "date": date,
        "commit_sha": _git_commit_sha(results_dir.resolve().parents[2])
                      if len(results_dir.resolve().parents) >= 3 else None,
        "glyphbench_version": _glyphbench_version(),
        "protocol": {
            "max_turns": "env-native",
            "seeds": "derived from np.random.default_rng(42)",
        },
    }
    meta_path = results_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    if tar_output is None:
        tar_output = results_dir.parent / f"{model_slug}__{harness}__{date}.tar.gz"

    with tarfile.open(tar_output, "w:gz") as tar:
        tar.add(results_dir, arcname=results_dir.name)

    return tar_output


def _cmd_bundle(args: argparse.Namespace) -> int:
    tar = _bundle_dir(args.results_dir, args.output)
    print(f"Bundle: {tar}")
    return 0


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def _discover_results_files(target: Path) -> list[Path]:
    """Return every results.jsonl under target. target is required."""
    if target.is_file() and target.name == "results.jsonl":
        return [target]
    if target.is_dir():
        return sorted(target.glob("**/results.jsonl"))
    return []


def _extract_block(content: str, header: str) -> str | None:
    """Extract a `[Header]\n…\n[Next]` block from a rendered user turn."""
    lines = content.split("\n")
    try:
        i = lines.index(header)
    except ValueError:
        return None
    out: list[str] = []
    for ln in lines[i + 1:]:
        if ln.startswith("[") and ln.endswith("]"):
            break
        out.append(ln)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out) if out else None


def _extract_grid(content: str) -> str | None:
    return _extract_block(content, "[Grid]")


_THINK_RE_LOCAL = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ACTION_RE_LOCAL = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
# Match any orphan <think>/<-/think> tag — Qwen3.5 chat template prefills
# <think>\n\n</think>\n\n so the assistant's own response often *starts*
# with a stray </think> (the closer of the prefill). Strip it cleanly.
_ORPHAN_THINK_TAG_RE = re.compile(r"</?\s*think\s*>", re.IGNORECASE)


def _split_assistant(content: str) -> tuple[str, str]:
    """Split an assistant turn into (reasoning, action). Reasoning falls back
    to anything-not-action if there is no <think> tag; action falls back to a
    bare uppercase token at the end (matching the parser's leniency)."""
    text = content or ""
    think = ""
    m = _THINK_RE_LOCAL.search(text)
    if m:
        think = m.group(1).strip()
    action = ""
    m = _ACTION_RE_LOCAL.search(text)
    if m:
        action = m.group(1).strip()
    if not think:
        residual = _ACTION_RE_LOCAL.sub("", text)
        residual = _THINK_RE_LOCAL.sub("", residual)
        # Strip any orphan <think> / </think> closers (chat-template prefill
        # leftovers) so they don't show up as standalone "reasoning".
        residual = _ORPHAN_THINK_TAG_RE.sub("", residual).strip()
        if residual:
            think = residual
    if not action:
        bare = re.findall(r"\b([A-Z][A-Z0-9_]{1,})\b", text)
        if bare:
            action = bare[-1]
    return think, action


def _model_from_jsonl_path(path: Path) -> str | None:
    """Recover the HF model id from `<root>/.../glyphbench--<owner>--<rest>/<hash>/results.jsonl`."""
    meta = path.parent / "metadata.json"
    if meta.exists():
        try:
            m = json.loads(meta.read_text()).get("model")
            if isinstance(m, str) and m:
                return m
        except (OSError, json.JSONDecodeError):
            pass
    for part in path.parts:
        if part.startswith("glyphbench--"):
            bits = part.split("--")
            if len(bits) >= 3:
                return "/".join([bits[1], "-".join(bits[2:])])
    return None


def _parse_info(info_field: object) -> dict:
    if isinstance(info_field, dict):
        return info_field
    if isinstance(info_field, str):
        try:
            return json.loads(info_field)
        except json.JSONDecodeError:
            return {}
    return {}


def _suite_of(env_id: str) -> str:
    return env_id.split("/", 1)[-1].split("-", 1)[0]


def _iter_rollouts(files: list[Path]) -> "list[tuple[Path, dict]]":
    """Yield (results_file, rollout_dict) for every parseable row."""
    out: list[tuple[Path, dict]] = []
    for f in files:
        try:
            with f.open() as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        out.append((f, json.loads(ln)))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return out


def _filter_rollouts(
    rollouts: list[tuple[Path, dict]],
    *,
    envs: list[str] | None,
    suites: list[str] | None,
    models: list[str] | None,
    seeds: list[int] | None,
) -> list[tuple[Path, dict, dict]]:
    """Return [(file, rollout, info)] passing every (AND-combined) filter."""
    out: list[tuple[Path, dict, dict]] = []
    env_set = set(envs) if envs else None
    suite_set = set(suites) if suites else None
    seed_set = set(seeds) if seeds else None
    model_set = set(models) if models else None
    for f, r in rollouts:
        info = _parse_info(r.get("info"))
        env_id = info.get("env_id", "")
        if env_set is not None and env_id not in env_set:
            continue
        if suite_set is not None and _suite_of(env_id) not in suite_set:
            continue
        if seed_set is not None:
            try:
                if int(info.get("seed")) not in seed_set:
                    continue
            except (TypeError, ValueError):
                continue
        if model_set is not None:
            model = _model_from_jsonl_path(f)
            if model not in model_set:
                continue
        out.append((f, r, info))
    return out


def _build_turns(rollout: dict) -> list[tuple[str, str]]:
    """Walk the rollout into [(user_content, assistant_content), …] in order."""
    prompt_msgs = rollout.get("prompt") or []
    completion = rollout.get("completion") or []
    initial_user = next((m for m in prompt_msgs if m.get("role") == "user"), None)
    turns: list[tuple[str, str]] = []
    i = 0
    if initial_user is not None and completion and completion[0].get("role") == "assistant":
        turns.append((initial_user.get("content") or "", completion[0].get("content") or ""))
        i = 1
    while i < len(completion):
        if completion[i].get("role") != "user":
            i += 1
            continue
        u = completion[i]
        a = completion[i + 1] if i + 1 < len(completion) and completion[i + 1].get("role") == "assistant" else None
        turns.append((u.get("content") or "", (a or {}).get("content") or ""))
        i += 2 if a is not None else 1
    return turns


def _scale_grid(grid: str, x: int, y: int) -> str:
    """Visually scale a glyph grid: each cell occupies `x` terminal columns
    (the glyph plus `x-1` trailing spaces) and `y` terminal rows (each line
    repeated `y` times). x=2,y=1 makes the cell roughly square in a typical
    2:1 height/width terminal font without doubling the glyph itself —
    `☺` becomes `☺ ` rather than `☺☺`."""
    if x <= 1 and y <= 1:
        return grid
    scaled_lines: list[str] = []
    pad = " " * max(0, x - 1)
    for line in grid.split("\n"):
        if x > 1:
            line = "".join(ch + pad for ch in line)
        for _ in range(max(1, y)):
            scaled_lines.append(line)
    return "\n".join(scaled_lines)


def _render_rollout_rich(
    rollout: dict,
    info: dict,
    model_id: str,
    delay: float,
    show_system: bool,
    pause: bool = False,
    grid_scale_x: int = 2,
    grid_scale_y: int = 1,
) -> None:
    """Render one rollout in a clean, in-place fullscreen layout: header
    bar at top, system prompt panel underneath (collapsible), then a per-turn
    composite (grid left, reasoning + action + env feedback right). Each turn
    fully replaces the previous frame — no scrollback churn."""
    import time

    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
    turns = _build_turns(rollout)
    if not turns:
        console.print(Text("(no turns to render)", style="red"))
        return

    # rollout-level flags we'll surface in the header on every frame
    overall_truncated = bool(rollout.get("is_truncated") or rollout.get("truncated_flag"))
    overall_completed = bool(rollout.get("is_completed") or rollout.get("terminated_flag"))
    overall_pf_rate = rollout.get("parse_failure_rate")
    stop_cond = rollout.get("stop_condition")
    reward = rollout.get("reward")
    metrics = rollout.get("metrics") or {}
    xml_ok = rollout.get("xml_format_reward", metrics.get("xml_format_reward"))

    sys_text = ""
    for m in rollout.get("prompt") or []:
        if m.get("role") == "system":
            sys_text = (m.get("content") or "").rstrip()
            break

    def render_frame(t_idx: int) -> Layout:
        u_content, a_content = turns[t_idx - 1]
        grid = _extract_grid(u_content) or "(no [Grid] block in this turn)"
        if grid_scale_x > 1 or grid_scale_y > 1:
            grid = _scale_grid(grid, grid_scale_x, grid_scale_y)
        hud = _extract_block(u_content, "[HUD]") or ""
        reward_block = _extract_block(u_content, "[Reward]") or ""
        status = _extract_block(u_content, "[Status]") or ""
        think, action = _split_assistant(a_content)

        # per-turn flags
        a_text = a_content or ""
        strict_action = bool(_ACTION_RE_LOCAL.search(a_text))
        strict_think = bool(_THINK_RE_LOCAL.search(a_text))
        max_tokens_hit = a_text and not strict_action and len(a_text) > 1500
        flags: list[Text] = []
        if not action:
            flags.append(Text(" PARSE FAIL ", style="bold white on red"))
        elif not strict_action:
            flags.append(Text(" parse-fallback ", style="bold black on yellow"))
        if not strict_think and think:
            flags.append(Text(" no <think> ", style="bold grey70 on grey23"))
        if max_tokens_hit:
            flags.append(Text(" likely token-truncated ", style="bold black on bright_yellow"))

        # ---- header bar ----
        hdr = Table.grid(expand=True, padding=(0, 1))
        hdr.add_column(justify="left", ratio=2)
        hdr.add_column(justify="right", ratio=1)
        left_pieces = [
            Text.assemble((model_id, "bold cyan"), " · ",
                          (info.get("env_id", "?"), "bold white")),
        ]
        if flags:
            for f in flags:
                left_pieces.append(Text(" ", end=""))
                left_pieces.append(f)
        right_pieces = Text.assemble(
            ("turn ", "dim"), (f"{t_idx}/{len(turns)}", "bold magenta"),
            "   ", ("seed=", "dim"), (str(info.get("seed", "?")), "yellow"),
            "   ", ("episodic reward=", "dim"), (f"{reward}", "bold green"),
            "   ", ("xml_ok=", "dim"), (f"{xml_ok:.2f}" if isinstance(xml_ok, (int, float)) else "?", "cyan"),
            "   ", ("pf_rate=", "dim"), (f"{overall_pf_rate:.2f}" if isinstance(overall_pf_rate, (int, float)) else "?", "yellow"),
        )
        hdr.add_row(Group(*left_pieces), right_pieces)

        rollout_state = []
        if overall_completed:
            rollout_state.append(Text(" terminated ", style="bold black on green"))
        if overall_truncated:
            rollout_state.append(Text(" truncated ", style="bold black on red"))
        if stop_cond:
            rollout_state.append(Text(f" stop={stop_cond} ", style="bold black on cyan"))
        if rollout_state:
            sub = Table.grid(expand=False, padding=(0, 1))
            for s in rollout_state:
                sub.add_column()
            sub.add_row(*rollout_state)
            header_block = Group(hdr, sub)
        else:
            header_block = hdr

        # ---- left: grid only ----
        grid_panel = Panel(
            Text(grid, style="bright_white"),
            title="grid",
            title_align="left",
            border_style="bright_white",
            padding=(0, 1),
        )

        # ---- right: HUD + reasoning + action + env feedback ----
        right_blocks: list = []
        if hud:
            right_blocks.append(
                Panel(Text(hud.strip(), style="cyan"),
                      title="HUD", title_align="left",
                      border_style="cyan", padding=(0, 1))
            )
        if think:
            right_blocks.append(
                Panel(Text(think.strip(), style="italic grey78"),
                      title="reasoning", title_align="left",
                      border_style="grey50", padding=(0, 1))
            )
        action_color = "green" if (action and strict_action) else ("yellow" if action else "red")
        right_blocks.append(
            Panel(
                Text(action or "(no action parsed)",
                     style=f"bold {action_color}"),
                title="action", title_align="left",
                border_style=action_color, padding=(0, 1),
            )
        )
        if reward_block or status:
            footer_lines: list = []
            if reward_block:
                footer_lines.append(Text.assemble(("reward: ", "bold"),
                                                  (reward_block.strip(), "yellow")))
            if status:
                footer_lines.append(Text.assemble(("status: ", "bold"),
                                                  (status.strip(), "magenta")))
            right_blocks.append(
                Panel(Group(*footer_lines), title="env feedback",
                      title_align="left", border_style="yellow", padding=(0, 1))
            )

        # ---- compose layout ----
        body = Layout(name="body")
        body.split_row(
            Layout(grid_panel, name="left", ratio=2),
            Layout(Group(*right_blocks), name="right", ratio=3),
        )

        root = Layout(name="root")
        sub_layouts = [Layout(Panel(header_block, border_style="cyan", padding=(0, 1)),
                              name="hdr", size=4 + (1 if rollout_state else 0))]
        if show_system and sys_text:
            sys_panel = Panel(
                Text(sys_text, style="dim", overflow="fold"),
                title="system prompt", title_align="left",
                border_style="grey39", padding=(0, 1),
            )
            sub_layouts.append(Layout(sys_panel, name="sys", size=8))
        sub_layouts.append(body)
        root.split_column(*sub_layouts)
        return root

    # Live: clean in-place screen (alternate buffer if TTY); each frame
    # fully replaces the previous one.
    is_tty = console.is_terminal
    with Live(render_frame(1), console=console, refresh_per_second=24,
              screen=is_tty, transient=False, redirect_stdout=False,
              redirect_stderr=False) as live:
        for t_idx in range(1, len(turns) + 1):
            live.update(render_frame(t_idx))
            if pause and is_tty:
                # Wait for a single keypress: any → next, q → quit rollout.
                ch = _read_one_key()
                if ch == "q":
                    break
            elif delay > 0:
                time.sleep(delay)
        if is_tty and not pause:
            # Hold the final frame visible so the user sees the outcome.
            time.sleep(min(2.0, max(0.5, delay * 8)))


def _read_one_key() -> str:
    """Read a single keypress from stdin without echo. Returns '' on EOF."""
    import sys
    try:
        import termios, tty
    except ImportError:
        # Non-POSIX (Windows): fall back to line input.
        try:
            return (sys.stdin.readline() or "").strip().lower()[:1]
        except Exception:
            return ""
    fd = sys.stdin.fileno()
    if not sys.stdin.isatty():
        return (sys.stdin.readline() or "").strip().lower()[:1]
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch.lower() if ch else ""


def _cmd_replay(args: argparse.Namespace) -> int:
    import time

    files = _discover_results_files(args.runs_dir)
    if not files:
        print(f"no results.jsonl found under {args.runs_dir}", file=sys.stderr)
        return 2

    rollouts = _iter_rollouts(files)
    if not rollouts:
        print(f"no rollouts in {len(files)} results.jsonl file(s)", file=sys.stderr)
        return 2

    filtered = _filter_rollouts(
        rollouts,
        envs=args.env,
        suites=args.suite,
        models=args.model,
        seeds=args.seed,
    )

    if args.list:
        rows: dict[tuple[str, str, object], int] = {}
        for f, _, info in filtered:
            key = (_model_from_jsonl_path(f) or "?", info.get("env_id", "?"), info.get("seed"))
            rows[key] = rows.get(key, 0) + 1
        if not rows:
            print("(no rollouts match the filter)", file=sys.stderr)
            return 1
        width_m = max(len(k[0]) for k in rows) + 2
        width_e = max(len(k[1]) for k in rows) + 2
        print(f"{'model':<{width_m}}{'env_id':<{width_e}}{'seed':>6}  rollouts")
        print("-" * (width_m + width_e + 16))
        for (m, e, s), n in sorted(rows.items()):
            print(f"{m:<{width_m}}{e:<{width_e}}{str(s):>6}  {n}")
        print(f"\n{len(filtered)} rollouts matched across {len(files)} file(s).")
        return 0

    if not filtered:
        print("(no rollouts match the filter; try `glyphbench replay <runs-dir> --list`)",
              file=sys.stderr)
        return 1

    if args.episode is not None:
        if args.episode < 0 or args.episode >= len(filtered):
            print(f"--episode {args.episode} out of range (0..{len(filtered) - 1})",
                  file=sys.stderr)
            return 2
        filtered = [filtered[args.episode]]
    elif args.limit is not None:
        filtered = filtered[: args.limit]

    if args.plain:
        clear = "\x1b[2J\x1b[H"
        for f, rollout, info in filtered:
            if not args.no_header:
                model = _model_from_jsonl_path(f) or "?"
                header = (f"=== model={model}  env={info.get('env_id', '?')}  "
                          f"seed={info.get('seed', '?')}  reward={rollout.get('reward')} ===")
                sys.stdout.write(clear + header + "\n\n")
                sys.stdout.flush()
                time.sleep(min(args.delay * 4, 1.0))
            for m in (list(rollout.get("prompt", [])) + list(rollout.get("completion", []))):
                if m.get("role") != "user":
                    continue
                grid = _extract_grid(m.get("content") or "")
                if grid is None:
                    continue
                sys.stdout.write(clear + grid + "\n")
                sys.stdout.flush()
                time.sleep(args.delay)
        return 0

    for f, rollout, info in filtered:
        model = _model_from_jsonl_path(f) or "?"
        _render_rollout_rich(
            rollout, info, model_id=model,
            delay=args.delay, show_system=not args.no_system,
            pause=args.pause,
            grid_scale_x=args.grid_scale_x,
            grid_scale_y=args.grid_scale_y,
        )
    return 0


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def _cmd_list_suites(args: argparse.Namespace) -> int:
    for s in ALL_SUITES:
        print(s)
    return 0


def _cmd_list_envs(args: argparse.Namespace) -> int:
    import glyphbench  # noqa: F401
    from glyphbench.core import all_glyphbench_env_ids

    envs = [e for e in all_glyphbench_env_ids() if "dummy" not in e]
    if args.suite:
        envs = [e for e in envs if f"/{args.suite}-" in e]
    for e in envs:
        print(e)
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("glyphbench", description="GlyphBench CLI")
    subs = p.add_subparsers(dest="cmd", required=True)

    pb = subs.add_parser("bundle", help="Package a results dir for submission")
    pb.add_argument("results_dir", type=Path,
                    help="Path to a results dir containing results.json")
    pb.add_argument("--output", type=Path, default=None,
                    help="Tarball path (default: sibling of results_dir).")
    pb.set_defaults(func=_cmd_bundle)

    pr = subs.add_parser("replay", help="Stream saved episodes as an animated grid")
    pr.add_argument("runs_dir", type=Path,
                    help="Dir tree containing results.jsonl files (e.g. "
                         "cluster_manager/results, or a single run-hash dir).")
    pr.add_argument("--env", action="append",
                    help="Filter by env_id (repeat for multiple, e.g. "
                         "--env glyphbench/atari-pong-v0). AND-combined with --suite/--model/--seed.")
    pr.add_argument("--suite", action="append",
                    help="Filter by suite name (repeat for multiple, e.g. --suite atari).")
    pr.add_argument("--model", action="append",
                    help="Filter by HF model id (repeat for multiple).")
    pr.add_argument("--seed", action="append", type=int,
                    help="Filter by seed integer (repeat for multiple).")
    pr.add_argument("--episode", type=int, default=None,
                    help="0-indexed pick from the filtered set. Plays only that one.")
    pr.add_argument("--limit", type=int, default=None,
                    help="Cap on rollouts to play after filtering.")
    pr.add_argument("--list", action="store_true",
                    help="Don't play anything — print the (model, env_id, seed) "
                         "index of what matches the filters and exit.")
    pr.add_argument("--plain", action="store_true",
                    help="Old grid-only renderer (no rich panels). Useful for "
                         "piping or sub-100-column terminals.")
    pr.add_argument("--no-system", action="store_true",
                    help="Suppress the per-rollout system-prompt panel.")
    pr.add_argument("--no-header", action="store_true",
                    help="(plain mode) suppress the per-rollout header banner.")
    pr.add_argument("--pause", action="store_true",
                    help="Step turn-by-turn: wait for a keypress between frames "
                         "(any key → next, q → next rollout). Overrides --delay.")
    pr.add_argument("--grid-scale-x", type=int, default=2, metavar="N",
                    dest="grid_scale_x",
                    help="Pad each glyph with N-1 trailing spaces so each "
                         "cell occupies N terminal columns. Default 2 "
                         "(~square cell on most terminals). 1 disables.")
    pr.add_argument("--grid-scale-y", type=int, default=1, metavar="N",
                    dest="grid_scale_y",
                    help="Repeat each grid row N times vertically. Default 1.")
    pr.add_argument("--delay", type=float, default=0.15,
                    help="Seconds between turns. Default: 0.15")
    pr.set_defaults(func=_cmd_replay)

    pls = subs.add_parser("list-suites", help="Print all suite names")
    pls.set_defaults(func=_cmd_list_suites)

    ple = subs.add_parser("list-envs", help="Print all env IDs")
    ple.add_argument("--suite", choices=ALL_SUITES,
                     help="Filter by a single suite")
    ple.set_defaults(func=_cmd_list_envs)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
