#!/usr/bin/env bash
# Single-node smoke — runs trainer + orchestrator + 7-way vLLM inference
# all on sf-node-1. Used while inter-node ports (8000, 29501) are blocked
# by the cluster firewall. Once those open, use launch_smoke_multinode.sh
# instead.
#
# This goes through prime-rl's `rl` entrypoint which spawns inference,
# orchestrator, and trainer as child processes on the same host. We
# inject our compute_advantages monkey-patch via a small shim before
# calling the rl main.

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR=${OUTPUT_DIR:-/home/roger/glyphbench/outputs/qwen35-4b-glyphbench-smoke-singlenode}
mkdir -p "$OUTPUT_DIR"

# Install our sitecustomize.py into the venv so prime-rl subprocesses
# can pick up the compute_advantages patch via the env-var hook below.
# The venv site-packages is rewritten by `uv sync`, so we copy fresh
# every launch (the file is checked into the repo at scripts/rl/).
SITE_PACKAGES="$REPO_ROOT/.venv/lib/python3.12/site-packages"
if [ -d "$SITE_PACKAGES" ]; then
    cp "$REPO_ROOT/scripts/rl/sitecustomize.py" "$SITE_PACKAGES/sitecustomize.py"
fi

# Tell sitecustomize.py to apply our patch in every python subprocess
# that prime-rl's `rl` entrypoint spawns (orchestrator, trainer,
# inference). Subprocesses inherit this env var.
export GLYPHBENCH_PATCH_PRIME_RL=1

echo "[$(hostname)] launching single-node smoke (rl entrypoint)"
echo "  config: configs/rl/qwen35-4b-glyphbench/smoke-singlenode.toml"
echo "  output: $OUTPUT_DIR"
echo "  GLYPHBENCH_PATCH_PRIME_RL=$GLYPHBENCH_PATCH_PRIME_RL"

uv run --extra rl --extra eval rl \
    @ configs/rl/qwen35-4b-glyphbench/smoke-singlenode.toml \
    --output-dir "$OUTPUT_DIR"
