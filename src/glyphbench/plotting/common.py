"""Data loaders and normalization for benchmark results.

These functions operate on the on-disk parquet format produced by RunStorage.
No Python imports from runner, providers, harness, or envs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file if it exists, else return empty DataFrame."""
    if path.exists():
        return pq.read_table(path).to_pandas()
    return pd.DataFrame()


def load_run(run_dir: str) -> dict[str, pd.DataFrame]:
    """Load a single run directory.

    Returns a dict with keys 'summary' and 'turns', each a DataFrame.
    Raises FileNotFoundError if the directory doesn't exist.
    """
    path = Path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    return {
        "summary": _read_parquet(path / "summary.parquet"),
        "turns": _read_parquet(path / "turns.parquet"),
    }


def load_runs(run_dirs: list[str]) -> pd.DataFrame:
    """Load and concatenate summary DataFrames from multiple runs.

    Adds a 'run_id' column derived from the directory name.
    Only reads summary.parquet (skips turns for efficiency).
    Returns an empty DataFrame if run_dirs is empty.
    """
    if not run_dirs:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        path = Path(run_dir)
        if not path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        summary = _read_parquet(path / "summary.parquet")
        if len(summary) > 0:
            summary = summary.copy()
            summary["run_id"] = path.name
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
    rand_means = random_baseline_summary.groupby("env_id")["episode_return"].mean()
    result = summary.copy()
    rand_mean_col = result["env_id"].map(rand_means).fillna(0.0)

    if expert_reference:
        ref_col = result["env_id"].map(expert_reference)
        denom = (ref_col - rand_mean_col).where(ref_col.notna(), 1.0)
    else:
        denom = pd.Series(1.0, index=result.index)

    denom = denom.replace(0.0, 1.0)
    result["normalized_return"] = (result["episode_return"] - rand_mean_col) / denom
    return result
