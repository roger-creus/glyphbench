"""Data loaders and normalization for benchmark results.

These functions operate on the on-disk parquet format produced by RunStorage.
No Python imports from runner, providers, harness, or envs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def load_run(run_dir: str) -> dict[str, pd.DataFrame]:
    """Load a single run directory.

    Returns a dict with keys 'summary' and 'turns', each a DataFrame.
    Raises FileNotFoundError if the directory doesn't exist.
    """
    path = Path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    result: dict[str, pd.DataFrame] = {}
    summary_path = path / "summary.parquet"
    if summary_path.exists():
        result["summary"] = pq.read_table(summary_path).to_pandas()
    else:
        result["summary"] = pd.DataFrame()

    turns_path = path / "turns.parquet"
    if turns_path.exists():
        result["turns"] = pq.read_table(turns_path).to_pandas()
    else:
        result["turns"] = pd.DataFrame()

    return result


def load_runs(run_dirs: list[str]) -> pd.DataFrame:
    """Load and concatenate summary DataFrames from multiple runs.

    Adds a 'run_id' column derived from the directory name.
    Returns an empty DataFrame if run_dirs is empty.
    """
    if not run_dirs:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        data = load_run(run_dir)
        summary = data["summary"]
        if len(summary) > 0:
            summary = summary.copy()
            summary["run_id"] = Path(run_dir).name
            frames.append(summary)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_normalized_scores(
    summary: pd.DataFrame,
    random_baseline_summary: pd.DataFrame,
    expert_reference: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add a 'normalized_return' column using random-baseline subtraction.

    Formula per row:
        normalized = (episode_return - random_mean) / (ref - random_mean)

    Where random_mean is the mean episode_return of the random baseline for
    that env_id, and ref is either the expert_reference value for that env_id
    (if provided) or 1.0 (identity scaling).
    """
    random_means = (
        random_baseline_summary.groupby("env_id")["episode_return"].mean().to_dict()
    )
    result = summary.copy()
    normalized: list[float] = []
    for _, row in result.iterrows():
        env_id = row["env_id"]
        ret = row["episode_return"]
        rand_mean = random_means.get(env_id, 0.0)
        if expert_reference and env_id in expert_reference:
            ref = expert_reference[env_id]
            denom = ref - rand_mean
        else:
            denom = 1.0
        if denom == 0.0:
            denom = 1.0
        normalized.append((ret - rand_mean) / denom)
    result["normalized_return"] = normalized
    return result
