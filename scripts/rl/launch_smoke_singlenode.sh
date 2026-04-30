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

# Point flashinfer at our user-space CUDA 12.6 toolkit. The system-wide
# /usr/bin/nvcc is CUDA 11.5 (too old for sm_90), and the cluster has no
# /usr/local/cuda. We extracted the cu12 toolchain into ~/cuda-12.6 from
# NVIDIA's conda packages (cuda-nvcc-tools, cuda-nvvm-tools/impl) plus
# the cccl + cuda-runtime headers from pip.
#
# Why 12.6 and not 12.8: the GPU driver here (565.57.01) tops out at
# CUDA 12.7. JIT-building flashinfer's gdn_prefill_sm90 with nvcc 12.8
# emitted host code that called TMA encoding APIs introduced in 12.8
# (cuTensorMapEncodeIm2colWide), and the 12.7 driver returned 999
# "Failed to initialize the TMA descriptor" at the first inference
# request. With nvcc 12.6, the kernel falls back to driver-supported
# TMA APIs and runs cleanly.
export CUDA_HOME=${CUDA_HOME:-/home/roger/cuda-12.6}
export PATH=$CUDA_HOME/bin:$PATH

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
