#!/usr/bin/env bash
# Start a vLLM inference server on a single node, with DP=8 (one worker per
# H100) and TP=1. Reads cluster config from .env.cluster (in repo root).
#
# Run on each inference node (sf-node-2, sf-node-3) — usually invoked via
# launch_all.sh which SSHes into them.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$REPO_ROOT/.env.cluster" ]; then
    echo "ERROR: $REPO_ROOT/.env.cluster not found. Copy from .env.cluster.template and fill in." >&2
    exit 2
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/.env.cluster"

: "${INFERENCE_PORT:?INFERENCE_PORT not set}"
: "${INFERENCE_SERVER_API_KEY:?INFERENCE_SERVER_API_KEY not set}"
: "${NCCL_SOCKET_IFNAME:?NCCL_SOCKET_IFNAME not set}"

export NCCL_SOCKET_IFNAME

if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
fi

mkdir -p "$REPO_ROOT/outputs/inference-logs"

# Read model name from the rl.toml. We don't parse TOML in bash; just hardcode
# matched to configs/rl/qwen35-4b-glyphbench/rl.toml::model.name. If you bump
# the model, update both places.
MODEL=Qwen/Qwen3.5-4B
MAX_MODEL_LEN=16384

echo "[$(hostname)] starting vLLM: model=$MODEL port=$INFERENCE_PORT dp=8 tp=1"

uv run --extra rl --extra eval inference \
    --model.name "$MODEL" \
    --model.max-model-len "$MAX_MODEL_LEN" \
    --server.port "$INFERENCE_PORT" \
    --server.api-key "$INFERENCE_SERVER_API_KEY" \
    --parallel.tp 1 \
    --parallel.dp 8 \
    --api-server-count 8 \
    --gpu-memory-utilization 0.85
