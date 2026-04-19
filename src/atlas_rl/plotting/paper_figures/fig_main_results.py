"""Main results: median normalized score per suite per model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from atlas_rl.plotting.common import load_runs
from atlas_rl.plotting.normalize import normalized_score
from atlas_rl.plotting.style import paper_style


def _suite_from_env_id(env_id: str) -> str:
    """Extract suite name from env_id like 'atlas_rl/atari-pong-v0'."""
    # env_id form: atlas_rl/<suite>-<rest>-v0
    # suite is the first dash-delimited segment after the slash
    after_slash = env_id.split("/")[-1]
    return after_slash.split("-")[0]


def generate(runs_dir: str, output_dir: str) -> None:
    """Bar chart of median normalized score per suite per model."""
    run_dirs = [str(p) for p in Path(runs_dir).iterdir() if p.is_dir()]
    if not run_dirs:
        return
    df = load_runs(run_dirs)
    if len(df) == 0:
        return

    df["normalized"] = df.apply(
        lambda r: normalized_score(r["env_id"], r["episode_return"]), axis=1
    )
    df = df.dropna(subset=["normalized"])
    df["suite"] = df["env_id"].apply(_suite_from_env_id)
    summary = df.groupby(["run_id", "suite"])["normalized"].median().reset_index()

    with paper_style():
        fig, ax = plt.subplots(figsize=(8, 4))
        if len(summary) > 0:
            pivot = summary.pivot(index="run_id", columns="suite", values="normalized")
            pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Median Normalized Score")
            ax.set_title("Main Results: Per-Suite Performance")
            ax.legend(title="Suite")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        fig.savefig(Path(output_dir) / "fig_main_results.pdf")
        fig.savefig(Path(output_dir) / "fig_main_results.png")
        plt.close(fig)
