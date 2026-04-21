"""Matplotlib style configuration for paper-ready plots."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

PAPER_STYLE: dict[str, Any] = {
    "figure.figsize": (5.5, 3.4),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["Computer Modern", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
}


def apply_style() -> None:
    """Apply PAPER_STYLE to global matplotlib rcParams."""
    mpl.rcParams.update(PAPER_STYLE)


@contextmanager
def paper_style() -> Generator[None, None, None]:
    """Context manager that temporarily applies PAPER_STYLE."""
    with plt.rc_context(PAPER_STYLE):
        yield
