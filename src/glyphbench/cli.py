"""GlyphBench CLI: wraps eval/run_eval.py for common flows + submission bundling.

Examples:
    glyphbench eval Qwen/Qwen3-8B
    glyphbench eval Qwen/Qwen3-8B --suite atari
    glyphbench eval Qwen/Qwen3-8B --env glyphbench/atari-pong-v0
    glyphbench eval Qwen/Qwen3-8B --harness all --bundle
    glyphbench bundle results/Qwen_Qwen3-8B/history_cot
    glyphbench list-suites
    glyphbench list-envs --suite atari
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

HARNESS_MODES = ["markov_zeroshot", "markov_cot", "history_zeroshot", "history_cot"]
ALL_SUITES = ["minigrid", "minihack", "atari", "procgen", "craftax", "classics"]


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def _slug_model(model: str) -> str:
    return model.replace("/", "_")


def _run_single_eval(args: argparse.Namespace, harness: str) -> Path:
    """Invoke eval/run_eval.py for one harness mode. Returns the output dir."""
    from eval import run_eval

    argv = [
        "--model", args.model,
        "--harness", harness,
        "--episodes", str(args.episodes),
        "--output", str(args.output),
    ]
    if args.suite:
        argv += ["--suites", *args.suite]
    if args.env:
        argv += ["--envs", *args.env]
    if args.history_len is not None:
        argv += ["--history-len", str(args.history_len)]
    if args.temperature is not None:
        argv += ["--temperature", str(args.temperature)]
    if args.batch_size is not None:
        argv += ["--batch-size", str(args.batch_size)]
    if args.extra:
        argv += args.extra

    run_eval.main(argv)
    return args.output / _slug_model(args.model) / harness


def _cmd_eval(args: argparse.Namespace) -> int:
    modes = HARNESS_MODES if args.harness == "all" else [args.harness]
    out_dirs: list[Path] = []
    for mode in modes:
        out = _run_single_eval(args, mode)
        out_dirs.append(out)
        if args.bundle:
            _bundle_dir(out, tar_output=None)
    return 0


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
    """Package one results dir (per-harness subdir) into a submission tarball.

    Expects results_dir to contain results.json + per_env/ + trajectories/
    (i.e., the directory produced by one harness mode of run_eval.py).
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

    pe = subs.add_parser("eval", help="Run an eval on a model")
    pe.add_argument("model", help="HF model ID (e.g., Qwen/Qwen3-8B)")
    scope = pe.add_mutually_exclusive_group()
    scope.add_argument("--suite", action="append", choices=ALL_SUITES,
                       help="One or more suites (repeat flag). Default: all suites.")
    scope.add_argument("--env", action="append",
                       help="One or more specific env IDs (repeat flag).")
    pe.add_argument("--harness", default="history_cot",
                    choices=[*HARNESS_MODES, "all"],
                    help="Harness mode (or 'all' for all 4 modes). Default: history_cot")
    pe.add_argument("--episodes", type=int, default=25)
    pe.add_argument("--history-len", type=int, default=None,
                    help="History window for history_* modes (passed through).")
    pe.add_argument("--temperature", type=float, default=None)
    pe.add_argument("--batch-size", type=int, default=None)
    pe.add_argument("--output", type=Path, default=Path("results"))
    pe.add_argument("--bundle", action="store_true",
                    help="Also produce a submission-ready tarball per harness.")
    pe.add_argument("--extra", nargs=argparse.REMAINDER,
                    help="Extra args passed verbatim to eval/run_eval.py")
    pe.set_defaults(func=_cmd_eval)

    pb = subs.add_parser("bundle", help="Package a results dir for submission")
    pb.add_argument("results_dir", type=Path,
                    help="Path to <output>/<model>/<harness>/ (contains results.json)")
    pb.add_argument("--output", type=Path, default=None,
                    help="Tarball path (default: sibling of results_dir).")
    pb.set_defaults(func=_cmd_bundle)

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
