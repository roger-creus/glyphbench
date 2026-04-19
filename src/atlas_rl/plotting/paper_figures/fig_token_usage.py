"""Token usage per model per suite."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from atlas_rl.plotting.common import load_runs
from atlas_rl.plotting.style import paper_style


def _suite_from_env_id(env_id: str) -> str:
    after_slash = env_id.split("/")[-1]
    return after_slash.split("-")[0]


def generate(runs_dir: str, output_dir: str) -> None:
    """Grouped bar chart of mean total tokens per (run_id, suite)."""
    run_dirs = [str(p) for p in Path(runs_dir).iterdir() if p.is_dir()]
    if not run_dirs:
        return
    df = load_runs(run_dirs)
    if len(df) == 0:
        return

    token_col = None
    for candidate in ("total_tokens", "tokens_used", "prompt_tokens"):
        if candidate in df.columns:
            token_col = candidate
            break

    with paper_style():
        fig, ax = plt.subplots(figsize=(8, 4))

        if token_col is not None:
            df["suite"] = df["env_id"].apply(_suite_from_env_id)
            summary = (
                df.groupby(["run_id", "suite"])[token_col].mean().reset_index()
            )
            if len(summary) > 0:
                pivot = summary.pivot(
                    index="run_id", columns="suite", values=token_col
                )
                pivot.plot(kind="bar", ax=ax)
                ax.set_ylabel("Mean Tokens per Episode")
                ax.legend(title="Suite")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        else:
            ax.text(
                0.5,
                0.5,
                "No token usage data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        ax.set_title("Token Usage by Model and Suite")
        fig.savefig(Path(output_dir) / "fig_token_usage.pdf")
        fig.savefig(Path(output_dir) / "fig_token_usage.png")
        plt.close(fig)
