"""Parse failure rates per model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from glyphbench.plotting.common import load_runs
from glyphbench.plotting.style import paper_style


def generate(runs_dir: str, output_dir: str) -> None:
    """Bar chart of action_parse_failure_rate per run_id."""
    run_dirs = [str(p) for p in Path(runs_dir).iterdir() if p.is_dir()]
    if not run_dirs:
        return
    df = load_runs(run_dirs)
    if len(df) == 0:
        return

    rate_col = None
    for candidate in ("action_parse_failure_rate", "parse_failure_rate", "parse_failures"):
        if candidate in df.columns:
            rate_col = candidate
            break

    with paper_style():
        fig, ax = plt.subplots(figsize=(6, 4))

        if rate_col is not None:
            rates = df.groupby("run_id")[rate_col].mean().sort_values(ascending=False)
            rates.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
            ax.set_ylabel("Mean Parse Failure Rate")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        else:
            ax.text(
                0.5,
                0.5,
                "No parse failure data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        ax.set_title("Action Parse Failure Rate by Model")
        fig.savefig(Path(output_dir) / "fig_parse_failures.pdf")
        fig.savefig(Path(output_dir) / "fig_parse_failures.png")
        plt.close(fig)
