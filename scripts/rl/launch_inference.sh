#!/usr/bin/env bash
# Start a vLLM inference server on a single node, with DP=8 (one worker per
# H100) and TP=1. Reads cluster config from .env.cluster (in repo root).
#
# Run on each inference node (sf-node-2, sf-node-3) — usually invoked via
# launch_all.sh which SSHes into them.

set -euo pipefail

# Non-interactive SSH does NOT source .bashrc, so $HOME/.local/bin (where
# `uv` lives by default) isn't on PATH. Prepend it explicitly.
export PATH="$HOME/.local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$REPO_ROOT/.env.cluster" ]; then
    echo "ERROR: $REPO_ROOT/.env.cluster not found. Copy from .env.cluster.template and fill in." >&2
    exit 2
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/.env.cluster"

: "${INFERENCE_PORT:?INFERENCE_PORT not set}"
: "${NCCL_SOCKET_IFNAME:?NCCL_SOCKET_IFNAME not set}"
# INFERENCE_SERVER_API_KEY is read but not used by prime-rl's inference
# entrypoint (which only exposes host/port). The vLLM router prime-rl ships
# proxies requests without an API-key check. Cluster security relies on
# network-level access control to ports 8000/29501 instead.
: "${INFERENCE_SERVER_API_KEY:=}"

export NCCL_SOCKET_IFNAME

if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
fi

mkdir -p "$REPO_ROOT/outputs/inference-logs"

# Model and max-context are env-overridable. Defaults match
# configs/rl/qwen35-4b-glyphbench/rl.toml (production). For the multi-node
# smoke, launch_smoke_multinode.sh sets MODEL=Qwen/Qwen3-4B-Instruct-2507
# and MAX_MODEL_LEN=8192 before calling launch_all.sh, which propagates to
# this script via SSH env.
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}

echo "[$(hostname)] starting vLLM: model=$MODEL port=$INFERENCE_PORT dp=8 tp=1"

uv run --extra rl --extra eval inference \
    --model.name "$MODEL" \
    --model.max-model-len "$MAX_MODEL_LEN" \
    --server.port "$INFERENCE_PORT" \
    --parallel.tp 1 \
    --parallel.dp 8 \
    --api-server-count 8 \
    --gpu-memory-utilization 0.85
