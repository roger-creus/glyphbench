#!/usr/bin/env bash
# Start the prime-rl trainer on the trainer node with FSDP across 8 GPUs.

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

# Trainer-specific override: prefer cuda-12.8 toolkit over the cluster
# default (12.6) because tilelang's gated-delta-rule chunk_bwd kernel
# emits TMA (`cp.async.bulk.tensor`) PTX whose state-space annotations
# nvcc 12.6's ptxas rejects with "State space incorrect for instruction
# 'cp.async.bulk.tensor'". nvcc 12.8 emits the correct PTX. The driver
# (565.57.01, CUDA 12.7 cap) accepts the resulting cubin so long as the
# kernel doesn't use 12.8-only PTX features beyond what 12.7 supports.
CUDA_HOME=${CUDA_HOME:-$HOME/cuda-12.8}
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    CUDA_HOME=$HOME/cuda-12.6
fi
if [ -x "$CUDA_HOME/bin/nvcc" ]; then
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
fi

# fla (flash-linear-attention) ships a tilelang backend for the gated
# delta-rule chunk bwd kernel. fla refuses to fall back to Triton on
# Hopper GPUs with Triton >= 3.4.0 (incorrect results, fla #640) — so
# tilelang is mandatory.
#
# tilelang's cuda_fp8.h references e8m0 FP8 APIs (__nv_fp8_e8m0,
# __nv_cvt_e8m0_*). The cu12==12.6.77 wheel that install_cuda_toolkit.sh
# extracts ships cuda_fp8.h without these (despite tilelang's gate
# saying ">=12.6"). install_cuda_toolkit.sh now overlays the 12.8
# cuda_fp8.{h,hpp} so nvcc 12.6 finds the e8m0 declarations and the
# tilelang JIT compiles successfully.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$REPO_ROOT/.env.cluster" ]; then
    echo "ERROR: $REPO_ROOT/.env.cluster not found." >&2
    exit 2
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/.env.cluster"

: "${OUTPUT_DIR:?OUTPUT_DIR not set}"
: "${NCCL_SOCKET_IFNAME:?NCCL_SOCKET_IFNAME not set}"

export NCCL_SOCKET_IFNAME

mkdir -p "$OUTPUT_DIR"

# Config is env-overridable for smoke / production swap.
CONFIG=${CONFIG:-configs/rl/qwen35-4b-glyphbench/rl.toml}

# Same split as launch_orchestrator.sh — TrainerConfig forbids extra
# top-level inputs from the unified RLConfig schema.
SPLIT_DIR=$(mktemp -d -t rl-split.XXXXXX)
trap 'rm -rf "$SPLIT_DIR"' EXIT
uv run --extra rl --extra eval python -m glyphbench.rl.split_rl_toml "$CONFIG" "$SPLIT_DIR" >/dev/null
TRAIN_CONFIG="$SPLIT_DIR/trainer.toml"

echo "[$(hostname)] starting trainer FSDP=8"
echo "  unified config: $CONFIG"
echo "  trainer subconfig: $TRAIN_CONFIG"
echo "  output dir: $OUTPUT_DIR"

# Resolve the trainer module's file path. Suppress stderr (gpt_oss
# modeling has a stray top-level [ERROR] print on import in this
# prime-rl version) and keep only the last stdout line so the captured
# value is always a clean path — torchrun chokes on multi-line args.
TRAIN_PY=$(uv run --extra rl --extra eval python -c 'import prime_rl.trainer.rl.train as m; print(m.__file__)' 2>/dev/null | tail -n1)
if [ ! -f "$TRAIN_PY" ]; then
    echo "ERROR: could not resolve prime_rl.trainer.rl.train file path (got '$TRAIN_PY')" >&2
    exit 2
fi
echo "  trainer module: $TRAIN_PY"

uv run --extra rl --extra eval torchrun \
    --nproc-per-node 8 \
    --local-ranks-filter 0 \
    "$TRAIN_PY" \
    @ "$TRAIN_CONFIG" \
    --output-dir "$OUTPUT_DIR"
