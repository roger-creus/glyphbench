"""Tests for plotting style configuration."""

from __future__ import annotations

import matplotlib as mpl


class TestPaperStyle:
    def test_paper_style_is_dict(self) -> None:
        from glyphbench.plotting.style import PAPER_STYLE
        assert isinstance(PAPER_STYLE, dict)

    def test_paper_style_keys(self) -> None:
        from glyphbench.plotting.style import PAPER_STYLE
        required_keys = [
            "figure.figsize", "figure.dpi", "savefig.dpi", "savefig.bbox",
            "font.family", "font.serif", "font.size", "axes.titlesize",
            "axes.labelsize", "axes.spines.top", "axes.spines.right",
            "axes.grid", "grid.alpha", "grid.linewidth", "legend.frameon",
            "legend.fontsize", "lines.linewidth",
        ]
        for key in required_keys:
            assert key in PAPER_STYLE, f"Missing key: {key}"

    def test_paper_style_values(self) -> None:
        from glyphbench.plotting.style import PAPER_STYLE
        assert PAPER_STYLE["figure.figsize"] == (5.5, 3.4)
        assert PAPER_STYLE["savefig.dpi"] == 300
        assert PAPER_STYLE["axes.spines.top"] is False
        assert PAPER_STYLE["axes.spines.right"] is False
        assert PAPER_STYLE["axes.grid"] is True
        assert PAPER_STYLE["font.family"] == "serif"
        assert PAPER_STYLE["legend.frameon"] is False

    def test_apply_style_sets_rcparams(self) -> None:
        from glyphbench.plotting.style import apply_style
        apply_style()
        assert mpl.rcParams["axes.spines.top"] is False
        assert mpl.rcParams["axes.spines.right"] is False
        assert mpl.rcParams["axes.grid"] is True

    def test_apply_style_context_manager(self) -> None:
        from glyphbench.plotting.style import paper_style
        old_val = mpl.rcParams["axes.spines.top"]
        with paper_style():
            assert mpl.rcParams["axes.spines.top"] is False
        assert mpl.rcParams["axes.spines.top"] == old_val
