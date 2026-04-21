#!/usr/bin/env python
# scripts/run_benchmark.py
"""CLI entry point for running a benchmark.

Usage:
    uv run python scripts/run_benchmark.py configs/examples/dummy_smoke.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from glyphbench.runner.config import RunConfig  # noqa: E402
from glyphbench.runner.runner import run_benchmark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an glyphbench benchmark")
    parser.add_argument("config_path", type=Path, help="Path to YAML run config")
    args = parser.parse_args()
    if not args.config_path.exists():
        print(f"config not found: {args.config_path}", file=sys.stderr)
        return 1
    config = RunConfig.from_yaml(args.config_path)
    asyncio.run(run_benchmark(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
