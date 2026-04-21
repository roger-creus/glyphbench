"""GlyphBench scoring: normalization, IQM aggregation, GlyphBench Score.

Usage:
    from eval.scoring import compute_glyphbench_scores
    scores = compute_glyphbench_scores(results_dir, baseline_path)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Suite definitions
# ---------------------------------------------------------------------------

SUITES = ["minigrid", "minihack", "atari", "procgen", "craftax", "classics"]

EPSILON = 1e-6  # Avoid division by zero

# Fraction of a suite a model may miss before we emit a coverage warning.
COVERAGE_WARN_THRESHOLD = 0.05


def env_to_suite(env_id: str) -> str:
    """Extract suite name from env ID (e.g., 'glyphbench/minigrid-empty-5x5-v0' -> 'minigrid')."""
    if "/" in env_id:
        return env_id.split("/")[1].split("-")[0]
    return "unknown"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_return(model_return: float, random_return: float) -> float:
    """Normalize a model's return relative to the random baseline.

    normalized = (model - random) / max(|random|, epsilon)

    Interpretation:
        0.0 = same as random
        1.0 = 2x better than random (if random > 0) or random-magnitude above 0
       >1.0 = significantly better than random
       <0.0 = worse than random
    """
    denominator = max(abs(random_return), EPSILON)
    return (model_return - random_return) / denominator


# ---------------------------------------------------------------------------
# IQM (Interquartile Mean)
# ---------------------------------------------------------------------------


def iqm(values: list[float]) -> float:
    """Interquartile Mean: mean of the middle 50% of values.

    Robust to outlier envs. Standard from Agarwal et al. 2021
    "Deep RL at the Edge of the Statistical Precipice".
    """
    if len(values) < 4:
        return float(np.mean(values)) if values else 0.0
    arr = np.sort(values)
    n = len(arr)
    q1_idx = n // 4
    q3_idx = 3 * n // 4
    return float(np.mean(arr[q1_idx:q3_idx]))


def iqm_with_ci(
    values: list[float], n_bootstrap: int = 10000, ci: float = 0.95,
) -> tuple[float, float, float]:
    """IQM with bootstrap confidence interval.

    Returns (iqm, ci_lower, ci_upper).
    """
    if len(values) < 4:
        m = float(np.mean(values)) if values else 0.0
        return m, m, m
    arr = np.array(values)
    rng = np.random.default_rng(42)
    bootstrap_iqms = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        bootstrap_iqms.append(iqm(sample.tolist()))
    alpha = (1 - ci) / 2
    lo = float(np.percentile(bootstrap_iqms, 100 * alpha))
    hi = float(np.percentile(bootstrap_iqms, 100 * (1 - alpha)))
    return iqm(values), lo, hi


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def load_random_baseline(path: Path) -> dict[str, float]:
    """Load random baseline and return {env_id: mean_return}."""
    with path.open() as f:
        data = json.load(f)
    return {
        env_id: entry["mean_return"]
        for env_id, entry in data.items()
        if isinstance(entry, dict) and "mean_return" in entry
    }


def load_model_results(results_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load all model results.

    Returns: {model_name: {harness: {env_id: result_dict}}}
    """
    models: dict[str, dict[str, dict[str, Any]]] = {}
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        models[model_name] = {}
        for harness_dir in sorted(model_dir.iterdir()):
            if not harness_dir.is_dir():
                continue
            harness = harness_dir.name
            per_env = harness_dir / "per_env"
            if not per_env.exists():
                continue
            envs = {}
            for f in per_env.glob("*.json"):
                d = json.loads(f.read_text())
                if "error" not in d and "env_id" in d:
                    envs[d["env_id"]] = d
            models[model_name][harness] = envs
    return models


def compute_glyphbench_scores(
    results_dir: Path,
    baseline_path: Path,
    harness: str | None = None,
) -> dict[str, Any]:
    """Compute the full GlyphBench scoring.

    Args:
        results_dir: Path to results/ directory with model subdirs.
        baseline_path: Path to random_baseline.json.
        harness: Which harness mode to score (default: best available).

    Returns dict with:
        - leaderboard: [{model, glyphbench_score, per_suite_iqm, ...}]
        - per_env: [{model, env_id, raw_return, normalized_return, ...}]
        - pairwise_winrate: {(model_a, model_b): win_fraction}
    """
    baseline = load_random_baseline(baseline_path)
    all_results = load_model_results(results_dir)

    # Pick harness mode
    if harness is None:
        # Use the harness that has the most data
        harness_counts: dict[str, int] = {}
        for model_data in all_results.values():
            for h, envs in model_data.items():
                harness_counts[h] = harness_counts.get(h, 0) + len(envs)
        harness = max(harness_counts, key=harness_counts.get) if harness_counts else "markov_zeroshot"

    # Expected env set = baseline keys (the source of truth for what exists).
    expected_envs = set(baseline.keys())
    expected_per_suite: dict[str, set[str]] = {s: set() for s in SUITES}
    for env_id in expected_envs:
        s = env_to_suite(env_id)
        if s in expected_per_suite:
            expected_per_suite[s].add(env_id)

    # Warn once per env: (a) missing-from-baseline, (b) near-zero baseline.
    warned_missing_baseline: set[str] = set()
    warned_near_zero: set[str] = set()

    # Compute per-env normalized scores
    per_env_rows: list[dict[str, Any]] = []
    model_suite_scores: dict[str, dict[str, list[float]]] = {}

    for model_name, model_data in all_results.items():
        if harness not in model_data:
            continue
        envs = model_data[harness]
        model_suite_scores[model_name] = {s: [] for s in SUITES}

        for env_id, result in envs.items():
            # (2) Skip envs not in baseline; warn once per unique env.
            if env_id not in baseline:
                if env_id not in warned_missing_baseline:
                    logger.warning(
                        "env %s not in random baseline; skipping from scoring "
                        "(cannot normalize without a reference)",
                        env_id,
                    )
                    warned_missing_baseline.add(env_id)
                continue

            random_return = baseline[env_id]

            # (3) Near-zero random baseline -> normalization noisy. Log once per env.
            if abs(random_return) < EPSILON and env_id not in warned_near_zero:
                logger.info(
                    "env %s has near-zero random baseline (|r|=%.2e); "
                    "normalization may be noisy",
                    env_id,
                    abs(random_return),
                )
                warned_near_zero.add(env_id)

            raw_return = result["mean_return"]
            norm = normalize_return(raw_return, random_return)
            suite = env_to_suite(env_id)

            per_env_rows.append({
                "model": model_name,
                "env_id": env_id,
                "suite": suite,
                "raw_return": raw_return,
                "random_return": random_return,
                "normalized_return": norm,
            })

            if suite in model_suite_scores[model_name]:
                model_suite_scores[model_name][suite].append(norm)

    # Compute per-suite IQM and overall GlyphBench Score
    leaderboard: list[dict[str, Any]] = []
    for model_name, suite_scores in model_suite_scores.items():
        suite_iqms = {}
        for suite in SUITES:
            scores = suite_scores.get(suite, [])
            if scores:
                val, lo, hi = iqm_with_ci(scores)
                suite_iqms[suite] = {"iqm": val, "ci_lo": lo, "ci_hi": hi, "n_envs": len(scores)}
            else:
                suite_iqms[suite] = {"iqm": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n_envs": 0}

        # GlyphBench Score = equal-weight mean of per-suite IQMs
        active_suites = [s for s in SUITES if suite_iqms[s]["n_envs"] > 0]
        glyphbench_score = (
            float(np.mean([suite_iqms[s]["iqm"] for s in active_suites]))
            if active_suites else 0.0
        )

        # Parse failure rate
        envs = all_results[model_name].get(harness, {})
        pfr_values = [e.get("parse_failure_rate", 0) for e in envs.values()]
        mean_pfr = float(np.mean(pfr_values)) if pfr_values else 0.0

        # (1) Per-suite coverage vs. expected baseline env set.
        scored_env_ids = {
            r["env_id"] for r in per_env_rows if r["model"] == model_name
        }
        coverage: dict[str, float] = {}
        for suite in SUITES:
            expected = expected_per_suite.get(suite, set())
            if not expected:
                # Suite has no envs in baseline -> coverage undefined; report 1.0.
                coverage[suite] = 1.0
                continue
            scored_in_suite = scored_env_ids & expected
            coverage[suite] = round(len(scored_in_suite) / len(expected), 4)

            missing_count = len(expected) - len(scored_in_suite)
            if missing_count / len(expected) > COVERAGE_WARN_THRESHOLD:
                logger.warning(
                    "model %s missing %d/%d envs in suite %s "
                    "(coverage=%.2f < %.2f); GlyphBench Score not directly "
                    "comparable to full-coverage models",
                    model_name,
                    missing_count,
                    len(expected),
                    suite,
                    coverage[suite],
                    1.0 - COVERAGE_WARN_THRESHOLD,
                )

        leaderboard.append({
            "model": model_name,
            "glyphbench_score": round(glyphbench_score, 4),
            "n_envs": sum(suite_iqms[s]["n_envs"] for s in SUITES),
            "mean_parse_failure_rate": round(mean_pfr, 4),
            "per_suite": suite_iqms,
            "coverage": coverage,
        })

    leaderboard.sort(key=lambda x: x["glyphbench_score"], reverse=True)

    # Pairwise win rate
    pairwise: dict[str, dict[str, float]] = {}
    models = list(model_suite_scores.keys())
    for i, m_a in enumerate(models):
        pairwise[m_a] = {}
        for j, m_b in enumerate(models):
            if i == j:
                pairwise[m_a][m_b] = 0.5
                continue
            # Count envs where m_a > m_b
            envs_a = {r["env_id"]: r["normalized_return"] for r in per_env_rows if r["model"] == m_a}
            envs_b = {r["env_id"]: r["normalized_return"] for r in per_env_rows if r["model"] == m_b}
            common = set(envs_a.keys()) & set(envs_b.keys())
            if not common:
                pairwise[m_a][m_b] = 0.5
                continue
            wins = sum(1 for e in common if envs_a[e] > envs_b[e])
            pairwise[m_a][m_b] = round(wins / len(common), 4)

    return {
        "harness": harness,
        "leaderboard": leaderboard,
        "per_env": per_env_rows,
        "pairwise_winrate": pairwise,
    }


def print_leaderboard(scores: dict[str, Any]) -> None:
    """Pretty-print the leaderboard."""
    print(f"\n{'='*80}")
    print(f"  GlyphBench Leaderboard  (harness: {scores['harness']})")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Model':<35} {'Score':>8} {'Envs':>6} {'PFR':>6}  ", end="")
    for s in SUITES:
        print(f"{s[:6]:>7}", end="")
    print()
    print("-" * 80 + "-" * (7 * len(SUITES)))

    for rank, entry in enumerate(scores["leaderboard"], 1):
        model = entry["model"][:34]
        score = entry["glyphbench_score"]
        n_envs = entry["n_envs"]
        pfr = entry["mean_parse_failure_rate"]
        print(f"{rank:<5} {model:<35} {score:>8.4f} {n_envs:>6} {pfr:>5.1%}  ", end="")
        for s in SUITES:
            val = entry["per_suite"][s]["iqm"]
            print(f"{val:>7.3f}", end="")
        print()

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute GlyphBench scores")
    parser.add_argument("--results", type=Path, default=Path("cluster_manager/results"))
    parser.add_argument("--baseline", type=Path, default=Path("eval/random_baseline.json"))
    parser.add_argument("--harness", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Save scores JSON")
    args = parser.parse_args()

    scores = compute_glyphbench_scores(args.results, args.baseline, args.harness)
    print_leaderboard(scores)

    if args.output:
        with args.output.open("w") as f:
            json.dump(scores, f, indent=2)
        print(f"Saved to {args.output}")
