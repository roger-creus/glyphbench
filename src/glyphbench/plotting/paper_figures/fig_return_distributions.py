"""Return distributions per environment, grouped by suite."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from glyphbench.plotting.common import load_runs
from glyphbench.plotting.style import paper_style


def _suite_from_env_id(env_id: str) -> str:
    after_slash = env_id.split("/")[-1]
    return after_slash.split("-")[0]


def generate(runs_dir: str, output_dir: str) -> None:
    """Violin plots of episode_return per env, one subplot per suite."""
    run_dirs = [str(p) for p in Path(runs_dir).iterdir() if p.is_dir()]
    if not run_dirs:
        return
    df = load_runs(run_dirs)
    if len(df) == 0:
        return

    df["suite"] = df["env_id"].apply(_suite_from_env_id)
    suites = sorted(df["suite"].unique())

    with paper_style():
        fig, axes = plt.subplots(
            1, len(suites), figsize=(4 * len(suites), 4), sharey=False
        )
        if len(suites) == 1:
            axes = [axes]

        for ax, suite in zip(axes, suites, strict=False):
            suite_df = df[df["suite"] == suite]
            env_ids = sorted(suite_df["env_id"].unique())
            data = [
                suite_df.loc[suite_df["env_id"] == eid, "episode_return"].values
                for eid in env_ids
            ]
            # Only plot envs that have data
            data = [d for d in data if len(d) > 0]
            labels = [eid.split("/")[-1] for eid, d in zip(env_ids, data, strict=False) if len(d) > 0]
            if data:
                ax.violinplot(data, positions=range(len(data)))
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_title(suite.capitalize())
            ax.set_ylabel("Episode Return")

        fig.suptitle("Return Distributions by Suite")
        fig.tight_layout()
        fig.savefig(Path(output_dir) / "fig_return_distributions.pdf")
        fig.savefig(Path(output_dir) / "fig_return_distributions.png")
        plt.close(fig)
