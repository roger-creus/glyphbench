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
import functools
import json
import re
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from glyphbench.verifiers_integration.memory import extract_memory_update
from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser

# Singleton parser instance used by the canonical eval pipeline; we route
# CLI replay action extraction through it so the displayed action matches
# what the verifiers eval actually scored. Spec is unavailable at replay
# time (we'd have to re-load the env), so spec-aware bare-name validation
# is best-effort: the parser returns the last candidate token, then the
# CLI applies a malformed-content cleanup pass to recover from things
# like ``ACTION_NAME=MOVE_FORWARD``.
_REPLAY_PARSER = GlyphbenchXMLParser()

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
    date = datetime.now(UTC).strftime("%Y-%m-%d")

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


_THINK_OPEN_RE = re.compile(r"<\s*think\s*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)
_THINK_RE_LOCAL = re.compile(
    r"<\s*think\s*>(.*?)<\s*/\s*think\s*>", re.DOTALL | re.IGNORECASE
)
_ACTION_RE_LOCAL = re.compile(
    r"<\s*action\s*>(.*?)<\s*/\s*action\s*>", re.DOTALL | re.IGNORECASE
)
_ACTION_OPEN_RE = re.compile(
    r"<\s*action\s*>(.*?)(?:<|\Z)", re.DOTALL | re.IGNORECASE
)
_BARE_NAME_RE = re.compile(r"\b([A-Z][A-Z0-9_]{1,})\b")


def _split_assistant(content: str) -> tuple[str, str]:
    """Split an assistant turn into (reasoning, action).

    Mirrors the tolerance of ``verifiers_integration.parser`` so the
    replay panel shows the same action the eval actually scored:

    * Reasoning is the segment ending at the LAST ``</think>``. Qwen3.5's
      chat template prefills ``<think>\\n``, so the stored content
      commonly starts mid-thinking with no opener; treat start-of-string
      as an implicit opener when only a closer is present.
    * Action is the LAST ``<action>…</action>`` (the model often quotes
      the template ``<action>ACTION_NAME</action>`` inside its CoT before
      emitting the real one — first-match would grab that placeholder).
      Fall back to an unclosed ``<action>…`` and finally to the LAST
      bare uppercase token.
    * If the captured action is malformed (``<``, ``=``, whitespace),
      pluck out the last bare-name token inside it — e.g. a model
      writing ``<action>ACTION_NAME=MOVE_FORWARD</action>`` should
      surface as ``MOVE_FORWARD``.
    """
    text = content or ""

    # ---- reasoning ----
    closes = list(_THINK_CLOSE_RE.finditer(text))
    if closes:
        last_close = closes[-1]
        opens_before = [
            m for m in _THINK_OPEN_RE.finditer(text)
            if m.end() <= last_close.start()
        ]
        start = opens_before[-1].end() if opens_before else 0
        think = text[start:last_close.start()].strip()
        post_think = text[last_close.end():]
    else:
        think = ""
        post_think = text

    # ---- action: route through the canonical eval parser, then apply
    # a CLI-side cleanup pass for malformed candidates (the parser
    # would have routed them to noop via spec validation; we don't have
    # a spec at replay time so we rescue the bare-name token).
    candidate = _REPLAY_PARSER._extract_candidate(text) or ""
    action = candidate.strip()
    if action and any(
        ch in action for ch in ("<", "=", "/", "?", "`", "\n", "\t", " ")
    ):
        bare = _BARE_NAME_RE.findall(action)
        if bare:
            action = bare[-1]

    # Reasoning residual fallback — only fires when there was no </think>
    # at all (genuine leak, not chat-template prefill).
    if not think and not closes:
        residual = _ACTION_RE_LOCAL.sub("", text)
        residual = _THINK_OPEN_RE.sub("", residual).strip()
        if residual:
            think = residual

    return think, action


@functools.lru_cache(maxsize=None)
def _spec_for_env_id(env_id: str) -> tuple[object, str] | None:
    """Look up (ActionSpec, noop_action_name) for an env id.

    Cached; returns ``None`` when the env can't be instantiated (e.g.
    optional native dep missing on this machine). Used by
    ``_resolve_action`` to canonicalise the displayed action through
    the same path the eval used.
    """
    try:
        from glyphbench.core.registry import make_env
        env = make_env(env_id)
        return env.action_spec, env.noop_action_name
    except Exception:
        return None


def _resolve_action(text: str, env_id: str | None) -> tuple[str, bool]:
    """Return (display_name, parse_failed) for an assistant message.

    When the env can be loaded, this calls ``GlyphbenchXMLParser.parse_action``
    with the env's ActionSpec — exactly the entry point the eval used —
    so the displayed action is the canonicalised name (or the env's
    noop on parse failure). Otherwise falls back to ``_split_assistant``
    which is the same parser fallback chain without spec validation.
    """
    raw = text or ""
    if env_id:
        spec_info = _spec_for_env_id(env_id)
        if spec_info is not None:
            spec, noop = spec_info
            try:
                _idx, canonical, failed = _REPLAY_PARSER.parse_action(
                    raw, spec, noop=noop,
                )
                return canonical, failed
            except Exception:
                pass
    _, action = _split_assistant(raw)
    return action, not bool(action)


def _clip_to_lines(
    text: str,
    max_lines: int,
    *,
    mode: str = "tail",
    max_line_width: int = 0,
) -> str:
    """Clip a text block to at most ``max_lines`` logical lines.

    ``mode`` is one of:
      * ``"tail"``   — keep the last lines (best for chain-of-thought,
        where the conclusion lives at the end).
      * ``"head"``   — keep the first lines.
      * ``"middle"`` — keep first half + last half with a "lines clipped"
        marker between them.

    When clipping happens the first/last/middle line of the result is a
    ``[... N lines clipped ...]`` marker, so the caller always knows how
    much was hidden. ``max_line_width`` truncates each LOGICAL line to
    that many characters (a defence against single-line walls of text
    that would otherwise wrap into many visual rows inside a Rich panel
    and overflow the layout). ``0`` disables per-line truncation.
    """
    raw = text or ""
    if max_lines <= 0:
        return ""
    lines = raw.splitlines()
    if max_line_width and max_line_width > 4:
        cut = max_line_width - 1
        lines = [
            (ln[:cut] + "…") if len(ln) > max_line_width else ln
            for ln in lines
        ]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    dropped = len(lines) - (max_lines - 1)
    marker = f"[… {dropped} lines clipped …]"
    if mode == "head":
        return "\n".join(lines[: max_lines - 1] + [marker])
    if mode == "middle":
        keep = max_lines - 1
        head = keep // 2
        tail = keep - head
        return "\n".join(lines[:head] + [marker] + (lines[-tail:] if tail else []))
    return "\n".join([marker] + lines[-(max_lines - 1):])


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


def _iter_rollouts(files: list[Path]) -> list[tuple[Path, dict]]:
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


def _content_from_role(messages: list[dict], role: str) -> str:
    msg = next((m for m in messages if m.get("role") == role), None)
    return (msg or {}).get("content") or ""


def _is_memory_update_user(msg: dict) -> bool:
    return (
        msg.get("role") == "user"
        and (msg.get("content") or "").lstrip().startswith("[Memory Update]")
    )


def _extract_prompt_memory(content: str) -> str:
    match = re.search(
        r"<\s*memory\s*>(.*?)<\s*/\s*memory\s*>",
        content or "",
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _attach_memory_update(turn: dict, memory_user: dict, memory_assistant: dict) -> None:
    memory_text = extract_memory_update(memory_assistant.get("content") or "").memory
    turn["memory"] = {
        "previous_memory": _extract_prompt_memory(turn.get("user") or ""),
        "stored_memory": memory_text,
        "memory_update_prompt": [memory_user],
        "memory_update_response": [memory_assistant],
    }


def _build_memory_turns(rollout: dict) -> list[dict] | None:
    trajectory = rollout.get("trajectory") or []
    turns: list[dict] = []
    saw_memory = False
    for step in trajectory:
        memory = (step.get("extras") or {}).get("glyphbench_memory")
        if not memory:
            continue
        saw_memory = True
        prompt = step.get("prompt") or []
        action_response = memory.get("action_response") or step.get("completion") or []
        turns.append(
            {
                "user": _content_from_role(prompt, "user"),
                "assistant": _content_from_role(action_response, "assistant"),
                "memory": memory,
            }
        )
    return turns if saw_memory else None


def _build_turns(rollout: dict) -> list[dict]:
    """Walk the rollout into per-environment-turn display records."""
    memory_turns = _build_memory_turns(rollout)
    if memory_turns is not None:
        return memory_turns

    prompt_msgs = rollout.get("prompt") or []
    completion = rollout.get("completion") or []
    initial_user = next((m for m in prompt_msgs if m.get("role") == "user"), None)
    turns: list[dict] = []
    i = 0
    if initial_user is not None and completion and completion[0].get("role") == "assistant":
        turns.append(
            {
                "user": initial_user.get("content") or "",
                "assistant": completion[0].get("content") or "",
                "memory": None,
            }
        )
        i = 1
    while i < len(completion):
        if completion[i].get("role") != "user":
            i += 1
            continue
        if _is_memory_update_user(completion[i]):
            if (
                turns
                and i + 1 < len(completion)
                and completion[i + 1].get("role") == "assistant"
            ):
                _attach_memory_update(turns[-1], completion[i], completion[i + 1])
                i += 2
            else:
                i += 1
            continue
        u = completion[i]
        a = completion[i + 1] if i + 1 < len(completion) and completion[i + 1].get("role") == "assistant" else None
        turns.append(
            {
                "user": u.get("content") or "",
                "assistant": (a or {}).get("content") or "",
                "memory": None,
            }
        )
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
    pause: bool = False,
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

    console_h = console.size.height or 40
    reasoning_cap = max(8, console_h // 4)
    memory_cap = max(5, console_h // 6)
    sys_cap = max(4, console_h // 8)
    line_width_cap = 200

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
        turn = turns[t_idx - 1]
        u_content = turn["user"]
        a_content = turn["assistant"]
        memory = turn.get("memory")
        grid = _extract_grid(u_content) or "(no [Grid] block in this turn)"
        # Fixed 2x1 visual scale: each glyph occupies 2 terminal columns
        # so the cell ends up roughly square in a typical 2:1 font.
        grid = _scale_grid(grid, 2, 1)
        hud = _extract_block(u_content, "[HUD]") or ""
        reward_block = _extract_block(u_content, "[Reward]") or ""
        status = _extract_block(u_content, "[Status]") or ""
        think, _ = _split_assistant(a_content)
        # Action is resolved through the canonical eval parser so the
        # displayed name matches what the eval scored (canonicalised
        # via ActionSpec when the env can be loaded; on parse failure
        # the env's noop name is shown, exactly as in eval rollouts).
        action, action_parse_failed = _resolve_action(a_content, info.get("env_id"))

        # per-turn flags
        a_text = a_content or ""
        strict_action = bool(_ACTION_RE_LOCAL.search(a_text))
        # Presence of </think> alone is sufficient: Qwen3.5's chat template
        # prefills the opener so it's almost never visible in the stored
        # assistant content.
        strict_think = bool(_THINK_CLOSE_RE.search(a_text))
        max_tokens_hit = a_text and not strict_action and len(a_text) > 1500
        flags: list[Text] = []
        if action_parse_failed:
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
            for _s in rollout_state:
                sub.add_column()
            sub.add_row(*rollout_state)
            header_block = Group(hdr, sub)
        else:
            header_block = hdr

        # Per-turn errors that should be highlighted in red on the
        # affected panel. ``action_parse_failed`` is authoritative — it
        # comes from the same parser the eval used, with spec validation
        # when available.
        action_failed = action_parse_failed
        action_fallback = bool(action) and not strict_action and not action_parse_failed
        think_missing = think and not strict_think  # raw text leaked instead of <think>
        think_absent = not think and not strict_action  # nothing parseable at all
        turn_has_error = action_failed or think_missing or think_absent or max_tokens_hit

        # ---- left: grid (red border if this turn has a generation error) ----
        grid_border = "red" if turn_has_error else "bright_white"
        grid_panel = Panel(
            Text(grid, style="bright_white"),
            title="grid" + ("  (turn error)" if turn_has_error else ""),
            title_align="left",
            border_style=grid_border,
            padding=(0, 1),
        )

        # ---- right: HUD + reasoning + action + env feedback (sized so
        # nothing in the column overflows the terminal). Each entry is
        # (panel, fixed-row-budget). The text inside is pre-clipped to
        # those budgets via _clip_to_lines, with mode="tail" for
        # reasoning + memory (the conclusion / latest update is what the
        # viewer cares about).
        right_entries: list[tuple[object, int]] = []
        if turn_has_error:
            err_msgs: list[str] = []
            if action_failed:
                err_msgs.append("no <action> parsed")
            elif action_fallback:
                err_msgs.append("action parsed via bare-name fallback")
            if think_missing:
                err_msgs.append("reasoning leaked outside <think>")
            if think_absent:
                err_msgs.append("no reasoning + no action")
            if max_tokens_hit:
                err_msgs.append("response likely truncated by max_tokens")
            right_entries.append((
                Panel(
                    Text("\n".join(err_msgs), style="bold white on red",
                         overflow="fold"),
                    title="turn errors", title_align="left",
                    border_style="red", padding=(0, 1),
                ),
                len(err_msgs) + 2,
            ))
        if memory and memory.get("previous_memory"):
            prev_txt = _clip_to_lines(
                str(memory.get("previous_memory", "")).strip(),
                memory_cap, mode="tail", max_line_width=line_width_cap,
            )
            right_entries.append((
                Panel(Text(prev_txt, style="cyan", overflow="fold"),
                      title="previous memory", title_align="left",
                      border_style="cyan", padding=(0, 1)),
                memory_cap + 2,
            ))
        if hud:
            hud_lines = max(2, min(6, hud.strip().count("\n") + 1))
            hud_txt = _clip_to_lines(hud.strip(), hud_lines, mode="tail",
                                     max_line_width=line_width_cap)
            right_entries.append((
                Panel(Text(hud_txt, style="cyan", overflow="fold"),
                      title="HUD", title_align="left",
                      border_style="cyan", padding=(0, 1)),
                hud_lines + 2,
            ))
        if think:
            reasoning_border = "red" if think_missing else "grey50"
            reasoning_title = "reasoning" + (" (no <think> tag)" if think_missing else "")
            think_txt = _clip_to_lines(
                think.strip(), reasoning_cap, mode="tail",
                max_line_width=line_width_cap,
            )
            right_entries.append((
                Panel(Text(think_txt, style="italic grey78", overflow="fold"),
                      title=reasoning_title, title_align="left",
                      border_style=reasoning_border, padding=(0, 1)),
                reasoning_cap + 2,
            ))
        action_color = "green" if (action and strict_action) else ("yellow" if action else "red")
        action_title = "action" + (
            " (PARSE FAIL)" if action_failed
            else " (fallback)" if action_fallback else ""
        )
        right_entries.append((
            Panel(
                Text(action or "(no action parsed)", style=f"bold {action_color}"),
                title=action_title, title_align="left",
                border_style=action_color, padding=(0, 1),
            ),
            3,
        ))
        if reward_block or status:
            footer_lines: list = []
            if reward_block:
                footer_lines.append(Text.assemble(("reward: ", "bold"),
                                                  (reward_block.strip(), "yellow")))
            if status:
                footer_lines.append(Text.assemble(("status: ", "bold"),
                                                  (status.strip(), "magenta")))
            right_entries.append((
                Panel(Group(*footer_lines), title="env feedback",
                      title_align="left", border_style="yellow", padding=(0, 1)),
                len(footer_lines) + 2,
            ))
        if memory is not None:
            stored_memory = str(memory.get("stored_memory") or "").strip()
            if stored_memory:
                stored_txt = _clip_to_lines(
                    stored_memory, memory_cap, mode="tail",
                    max_line_width=line_width_cap,
                )
                right_entries.append((
                    Panel(Text(stored_txt, style="bright_cyan", overflow="fold"),
                          title="updated memory", title_align="left",
                          border_style="bright_cyan", padding=(0, 1)),
                    memory_cap + 2,
                ))

        # ---- compose layout ----
        body = Layout(name="body")
        body.split_row(
            Layout(grid_panel, name="left", ratio=2),
            Layout(name="right", ratio=3),
        )
        body["right"].split_column(
            *[Layout(panel, size=size) for panel, size in right_entries]
        )

        root = Layout(name="root")
        sub_layouts = [Layout(Panel(header_block, border_style="cyan", padding=(0, 1)),
                              name="hdr", size=4 + (1 if rollout_state else 0))]
        if sys_text:
            clipped_sys = _clip_to_lines(
                sys_text, sys_cap, mode="head",
                max_line_width=line_width_cap,
            )
            sys_panel = Panel(
                Text(clipped_sys, style="dim", overflow="fold"),
                title="system prompt", title_align="left",
                border_style="grey39", padding=(0, 1),
            )
            sub_layouts.append(Layout(sys_panel, name="sys", size=sys_cap + 2))
        sub_layouts.append(body)
        root.split_column(*sub_layouts)
        return root

    # Live: clean in-place screen (alternate buffer if TTY); each frame
    # fully replaces the previous one. auto_refresh=False — we drive
    # refreshes ourselves via live.update(..., refresh=True), so the
    # background ticker can't race with the raw-mode stdin read in
    # pause-mode (which would otherwise cause flickering / partial
    # frames after each keypress).
    is_tty = console.is_terminal
    with Live(render_frame(1), console=console, auto_refresh=False,
              screen=is_tty, transient=False, redirect_stdout=False,
              redirect_stderr=False) as live:
        for t_idx in range(1, len(turns) + 1):
            live.update(render_frame(t_idx), refresh=True)
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
        import termios
        import tty
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

    use_rich = sys.stdout.isatty()
    if not use_rich:
        clear = "\x1b[2J\x1b[H"
        for f, rollout, info in filtered:
            model = _model_from_jsonl_path(f) or "?"
            header = (f"=== model={model}  env={info.get('env_id', '?')}  "
                      f"seed={info.get('seed', '?')}  reward={rollout.get('reward')} ===")
            sys.stdout.write(clear + header + "\n\n")
            sys.stdout.flush()
            time.sleep(min(args.delay * 4, 1.0))
            for turn in _build_turns(rollout):
                grid = _extract_grid(turn["user"])
                if grid is None:
                    continue
                memory = turn.get("memory")
                suffix = ""
                if memory and memory.get("stored_memory"):
                    suffix = "\n\n[Memory]\n" + _clip_to_lines(
                        str(memory["stored_memory"]).strip(),
                        8, mode="tail", max_line_width=200,
                    )
                sys.stdout.write(clear + grid + suffix + "\n")
                sys.stdout.flush()
                time.sleep(args.delay)
        return 0

    for f, rollout, info in filtered:
        model = _model_from_jsonl_path(f) or "?"
        _render_rollout_rich(
            rollout, info, model_id=model,
            delay=args.delay,
            pause=args.pause,
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

    pr = subs.add_parser(
        "replay",
        help="Animate saved rollouts (rich panels in a TTY, plain text "
             "otherwise). Memory mode is rendered automatically when the "
             "rollout was produced with use_memory=True.",
    )
    pr.add_argument("runs_dir", type=Path,
                    help="Dir tree containing results.jsonl files (e.g. "
                         "cluster_manager/results, or a single run-hash dir).")
    pr.add_argument("--env", action="append",
                    help="Filter by env_id. Repeatable. AND-combined with "
                         "--suite/--model/--seed.")
    pr.add_argument("--suite", action="append",
                    help="Filter by suite name. Repeatable.")
    pr.add_argument("--model", action="append",
                    help="Filter by HF model id. Repeatable.")
    pr.add_argument("--seed", action="append", type=int,
                    help="Filter by seed integer. Repeatable.")
    pr.add_argument("--episode", type=int, default=None,
                    help="0-indexed pick from the filtered set; plays only that one.")
    pr.add_argument("--list", action="store_true",
                    help="Don't play anything; print the (model, env_id, seed) "
                         "index of what matches the filters and exit.")
    pr.add_argument("--pause", action="store_true",
                    help="Step turn-by-turn: any key → next frame, q → skip "
                         "to next rollout. Overrides --delay.")
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
