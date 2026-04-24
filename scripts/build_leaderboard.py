#!/usr/bin/env python
"""Build `docs/leaderboard/data.json` from `results/` directory.

Each results subdirectory is one (model, harness) run and contains:
  results/<model>__<harness>/
    results.json                # aggregate metrics
    per_env/*.json              # per-env metrics

This script aggregates them into the JSON consumed by docs/leaderboard/index.html.

Usage:
    uv run python scripts/build_leaderboard.py --results results/ --output docs/leaderboard/data.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from glyphbench.core import all_glyphbench_env_ids  # type: ignore[import-not-found]


SUITES = ("minigrid", "minihack", "atari", "procgen", "craftax", "classics")


def load_baseline(baseline_path: Path) -> dict[str, float]:
    data = json.loads(baseline_path.read_text())
    return {
        eid: info["mean_return"]
        for eid, info in data.items()
        if isinstance(info, dict) and "mean_return" in info
    }


def normalize(env_id: str, mean_return: float, baseline: dict[str, float]) -> float:
    """Random-normalized score: (return - random) / |random| clipped to [-1, 10]."""
    base = baseline.get(env_id, 0.0)
    denom = max(abs(base), 1.0)
    return max(-1.0, min(10.0, (mean_return - base) / denom))


def iqm(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    lo, hi = n // 4, (3 * n) // 4
    iqr = arr[lo:hi] if hi > lo else arr
    return sum(iqr) / len(iqr) if iqr else 0.0


def suite_of(env_id: str) -> str:
    name = env_id.split("/", 1)[1]
    return name.split("-", 1)[0]


def aggregate_run(run_dir: Path, baseline: dict[str, float]) -> dict | None:
    """Aggregate all per-env JSONs under a run dir. The run dir may hold a
    partial `results.json` (only one suite, when jobs are split by suite), so
    we prefer the full `per_env/*.json` set for scoring and fall back to `meta`
    only for model/harness identity."""
    per_env_dir = run_dir / "per_env"
    per_env_files = sorted(per_env_dir.glob("*.json")) if per_env_dir.exists() else []
    if not per_env_files:
        return None
    results_json = run_dir / "results.json"
    meta = {}
    if results_json.exists():
        raw = results_json.read_text().strip()
        if raw:
            try:
                meta = json.loads(raw)
            except json.JSONDecodeError:
                meta = {}

    per_suite_scores: dict[str, list[float]] = {s: [] for s in SUITES}
    total_input_tokens = 0
    total_output_tokens = 0
    parse_fails = 0
    total_steps = 0

    for f in per_env_files:
        raw = f.read_bytes().rstrip(b"\x00").decode("utf-8", errors="replace").strip()
        if not raw:
            continue
        try:
            info = json.loads(raw)
        except json.JSONDecodeError:
            continue
        env_id = info["env_id"]
        suite = suite_of(env_id)
        if suite not in per_suite_scores:
            continue
        norm = normalize(env_id, info["mean_return"], baseline)
        per_suite_scores[suite].append(norm)
        total_input_tokens += info.get("total_input_tokens", 0)
        total_output_tokens += info.get("total_output_tokens", 0)
        parse_fails += int(info.get("mean_parse_failures", 0) * info["n_episodes"])
        total_steps += int(info["mean_length"] * info["n_episodes"])

    per_suite = {s: iqm(v) for s, v in per_suite_scores.items() if v}
    overall = sum(per_suite.values()) / len(per_suite) if per_suite else 0.0

    return {
        "id": run_dir.name,
        "model": meta.get("model", run_dir.name.split("__")[0]),
        "harness": meta.get("harness", "unknown"),
        "score": overall,
        "per_suite": per_suite,
        "parse_failure_rate": parse_fails / max(total_steps, 1),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost": meta.get("total_cost"),
        "n_envs": len(per_env_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--baseline", type=Path, default=Path("eval/random_baseline.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/leaderboard/data.json"))
    args = parser.parse_args()

    baseline = load_baseline(args.baseline)

    runs = []
    if args.results.exists():
        # Two layouts supported:
        # 1) results/<slug>/results.json               (flat; slug is the run id)
        # 2) results/<model>/<harness>/results.json    (nested; cluster manager layout)
        for model_dir in sorted(args.results.iterdir()):
            if not model_dir.is_dir() or model_dir.name == "logs":
                continue
            if (model_dir / "results.json").exists():
                agg = aggregate_run(model_dir, baseline)
                if agg:
                    runs.append(agg)
            else:
                for harness_dir in sorted(model_dir.iterdir()):
                    if not harness_dir.is_dir():
                        continue
                    agg = aggregate_run(harness_dir, baseline)
                    if agg:
                        # Always override identity from directory names: meta
                        # comes from a single-suite sub-job and may be incomplete.
                        agg["model"] = model_dir.name.replace("_", "/", 1)
                        agg["harness"] = harness_dir.name
                        agg["id"] = f"{model_dir.name}__{harness_dir.name}"
                        runs.append(agg)

    data = {
        "generated_at": dt.date.today().isoformat(),
        "schema_version": 1,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2))
    print(f"Wrote {len(runs)} runs to {args.output}")


if __name__ == "__main__":
    main()
