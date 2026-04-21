#!/usr/bin/env python3
"""Foundation smoke analysis -- produces 7 paper-ready plots from a benchmark run.

Usage:
    python -m glyphbench.plotting.notebooks.foundation_smoke <run_dir>

Plots saved to <run_dir>/figures/*.{pdf,png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]

from glyphbench.plotting.common import load_run
from glyphbench.plotting.style import paper_style


def _save_fig(fig: matplotlib.figure.Figure, fig_dir: Path, name: str) -> None:
    """Save figure as both PDF and PNG."""
    fig.savefig(fig_dir / f"{name}.pdf")
    fig.savefig(fig_dir / f"{name}.png")
    plt.close(fig)


def _empty_plot(fig_dir: Path, name: str, title: str) -> None:
    """Save an empty plot as placeholder when no data is available."""
    fig, ax = plt.subplots()
    ax.set_title(title)
    _save_fig(fig, fig_dir, name)


def plot_return_distribution(summary: pd.DataFrame, fig_dir: Path) -> None:
    """1. Per-env return distribution (violin plot)."""
    fig, ax = plt.subplots()
    if len(summary) > 0:
        sns.violinplot(data=summary, x="env_id", y="episode_return", ax=ax, cut=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Return Distribution per Environment")
    ax.set_xlabel("Environment")
    ax.set_ylabel("Episode Return")
    _save_fig(fig, fig_dir, "return_distribution")


def plot_episode_length(summary: pd.DataFrame, fig_dir: Path) -> None:
    """2. Episode length distribution (box plot)."""
    fig, ax = plt.subplots()
    if len(summary) > 0:
        sns.boxplot(data=summary, x="env_id", y="episode_length", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Episode Length Distribution")
    ax.set_xlabel("Environment")
    ax.set_ylabel("Episode Length (turns)")
    _save_fig(fig, fig_dir, "episode_length")


def plot_token_usage(summary: pd.DataFrame, fig_dir: Path) -> None:
    """3. Token usage (grouped bar chart)."""
    if len(summary) == 0:
        _empty_plot(fig_dir, "token_usage", "Token Usage per Environment")
        return
    means = summary.groupby("env_id")[["total_tokens_in", "total_tokens_out"]].mean()
    fig, ax = plt.subplots()
    means.plot(kind="bar", ax=ax)
    ax.set_title("Mean Token Usage per Environment")
    ax.set_xlabel("Environment")
    ax.set_ylabel("Tokens")
    ax.legend(["Input", "Output"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    _save_fig(fig, fig_dir, "token_usage")


def plot_parse_failure_rate(summary: pd.DataFrame, fig_dir: Path) -> None:
    """4. Parse failure rate (bar chart)."""
    if len(summary) == 0:
        _empty_plot(fig_dir, "parse_failure_rate", "Action Parse Failure Rate")
        return
    means = summary.groupby("env_id")["action_parse_failure_rate"].mean()
    fig, ax = plt.subplots()
    means.plot(kind="bar", ax=ax)
    ax.set_title("Action Parse Failure Rate per Environment")
    ax.set_xlabel("Environment")
    ax.set_ylabel("Failure Rate")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    _save_fig(fig, fig_dir, "parse_failure_rate")


def plot_action_distribution(turns: pd.DataFrame, fig_dir: Path) -> None:
    """5. Action distribution (stacked bar of top-10 actions per env)."""
    if len(turns) == 0 or "action_name" not in turns.columns:
        _empty_plot(fig_dir, "action_distribution", "Action Distribution")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    action_counts = turns.groupby(["env_id", "action_name"]).size().unstack(fill_value=0)
    top_actions = action_counts.sum().nlargest(10).index
    action_counts = action_counts[top_actions]
    action_counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Top-10 Action Distribution per Environment")
    ax.set_xlabel("Environment")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Action", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    _save_fig(fig, fig_dir, "action_distribution")


def plot_cost_per_episode(summary: pd.DataFrame, fig_dir: Path) -> None:
    """6. Cost per episode (bar chart)."""
    if len(summary) == 0:
        _empty_plot(fig_dir, "cost_per_episode", "Cost per Episode")
        return
    means = summary.groupby("env_id")["total_dollar_cost"].mean()
    fig, ax = plt.subplots()
    means.plot(kind="bar", ax=ax)
    ax.set_title("Mean Cost per Episode (USD)")
    ax.set_xlabel("Environment")
    ax.set_ylabel("Cost ($)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    _save_fig(fig, fig_dir, "cost_per_episode")


def plot_return_over_episodes(summary: pd.DataFrame, fig_dir: Path) -> None:
    """7. Return over episodes (learning-curve style)."""
    if len(summary) == 0:
        _empty_plot(fig_dir, "return_over_episodes", "Return over Episodes")
        return
    fig, ax = plt.subplots()
    for env_id in sorted(summary["env_id"].unique()):
        env_data = summary[summary["env_id"] == env_id].sort_values("episode_idx")
        ax.plot(
            env_data["episode_idx"], env_data["episode_return"],
            label=env_id, marker="o", markersize=3,
        )
    ax.set_title("Return over Episodes")
    ax.set_xlabel("Episode Index")
    ax.set_ylabel("Episode Return")
    ax.legend(fontsize=7)
    _save_fig(fig, fig_dir, "return_over_episodes")


def main(run_dir: str) -> None:
    """Generate all 7 plots for a run directory."""
    data = load_run(run_dir)
    summary = data["summary"]
    turns = data["turns"]

    fig_dir = Path(run_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with paper_style():
        plot_return_distribution(summary, fig_dir)
        plot_episode_length(summary, fig_dir)
        plot_token_usage(summary, fig_dir)
        plot_parse_failure_rate(summary, fig_dir)
        plot_action_distribution(turns, fig_dir)
        plot_cost_per_episode(summary, fig_dir)
        plot_return_over_episodes(summary, fig_dir)

    print(f"Saved 7 plots to {fig_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <run_dir>")
        sys.exit(1)
    main(sys.argv[1])
