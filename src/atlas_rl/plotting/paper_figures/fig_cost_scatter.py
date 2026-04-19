"""Cost vs normalized score scatter plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from atlas_rl.plotting.common import load_runs
from atlas_rl.plotting.normalize import normalized_score
from atlas_rl.plotting.style import paper_style


def _suite_from_env_id(env_id: str) -> str:
    after_slash = env_id.split("/")[-1]
    return after_slash.split("-")[0]


def generate(runs_dir: str, output_dir: str) -> None:
    """Scatter: mean dollar cost (x) vs median normalized score (y), per (model, suite)."""
    run_dirs = [str(p) for p in Path(runs_dir).iterdir() if p.is_dir()]
    if not run_dirs:
        return
    df = load_runs(run_dirs)
    if len(df) == 0:
        return

    # normalized score
    df["normalized"] = df.apply(
        lambda r: normalized_score(r["env_id"], r["episode_return"]), axis=1
    )
    df = df.dropna(subset=["normalized"])
    df["suite"] = df["env_id"].apply(_suite_from_env_id)

    # cost column may be named cost_usd or total_cost; skip if absent
    cost_col = None
    for candidate in ("cost_usd", "total_cost", "cost"):
        if candidate in df.columns:
            cost_col = candidate
            break

    with paper_style():
        fig, ax = plt.subplots(figsize=(6, 4))

        if cost_col is not None:
            grouped = df.groupby(["run_id", "suite"]).agg(
                mean_cost=(cost_col, "mean"),
                median_normalized=("normalized", "median"),
            ).reset_index()

            suites = sorted(grouped["suite"].unique())
            for suite in suites:
                sub = grouped[grouped["suite"] == suite]
                ax.scatter(
                    sub["mean_cost"],
                    sub["median_normalized"],
                    label=suite,
                    s=60,
                    alpha=0.8,
                )
            ax.set_xlabel("Mean Episode Cost (USD)")
            ax.legend(title="Suite", fontsize=8)
        else:
            ax.text(
                0.5,
                0.5,
                "No cost data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        ax.set_ylabel("Median Normalized Score")
        ax.set_title("Cost vs Performance")
        fig.savefig(Path(output_dir) / "fig_cost_scatter.pdf")
        fig.savefig(Path(output_dir) / "fig_cost_scatter.png")
        plt.close(fig)
