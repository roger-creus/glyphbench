#!/usr/bin/env bash
# Start the prime-rl trainer on the trainer node with FSDP across 8 GPUs.

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

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
: "${TRAINER_NODE_IP:?TRAINER_NODE_IP not set}"
: "${NCCL_PORT:?NCCL_PORT not set}"

export NCCL_SOCKET_IFNAME

mkdir -p "$OUTPUT_DIR"

# Config is env-overridable for smoke / production swap.
CONFIG=${CONFIG:-configs/rl/qwen35-4b-glyphbench/rl.toml}

echo "[$(hostname)] starting trainer FSDP=8"
echo "  config: $CONFIG"
echo "  output dir: $OUTPUT_DIR"

uv run --extra rl --extra eval torchrun \
    --nproc-per-node 8 \
    --local-ranks-filter 0 \
    "$(uv run --extra rl python -c 'import prime_rl.trainer.rl.train as m; print(m.__file__)')" \
    @ "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --weight-broadcast.host "$TRAINER_NODE_IP" \
    --weight-broadcast.port "$NCCL_PORT"
