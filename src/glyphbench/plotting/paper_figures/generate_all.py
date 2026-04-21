#!/usr/bin/env python3
"""Generate all paper figures from benchmark runs.

Usage: python -m glyphbench.plotting.paper_figures.generate_all <runs_dir> <output_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from glyphbench.plotting.paper_figures.fig_cost_scatter import generate as gen_cost
from glyphbench.plotting.paper_figures.fig_main_results import generate as gen_main
from glyphbench.plotting.paper_figures.fig_parse_failures import generate as gen_parse
from glyphbench.plotting.paper_figures.fig_return_distributions import (
    generate as gen_returns,
)
from glyphbench.plotting.paper_figures.fig_token_usage import generate as gen_tokens


def main(runs_dir: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    generators = [gen_main, gen_returns, gen_cost, gen_parse, gen_tokens]
    for gen in generators:
        try:
            gen(runs_dir, str(out))
            print(f"  Generated: {gen.__module__.split('.')[-1]}")
        except Exception as e:
            print(f"  SKIPPED {gen.__module__.split('.')[-1]}: {e}")

    print(f"\nFigures saved to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <runs_dir> <output_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
