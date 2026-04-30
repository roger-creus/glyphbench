#!/usr/bin/env bash
# Start the prime-rl orchestrator on the trainer node, with our
# compute_advantages monkey-patch installed.

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

: "${INFERENCE_NODES:?INFERENCE_NODES not set}"
: "${INFERENCE_PORT:?INFERENCE_PORT not set}"
: "${OUTPUT_DIR:?OUTPUT_DIR not set}"
: "${NCCL_SOCKET_IFNAME:?NCCL_SOCKET_IFNAME not set}"
: "${TRAINER_NODE_IP:?TRAINER_NODE_IP not set}"
: "${NCCL_PORT:?NCCL_PORT not set}"
: "${INFERENCE_SERVER_API_KEY:=}"

# Set OPENAI_API_KEY_INFERENCE (used by prime-rl's client.api-key-var) — empty
# is OK because prime-rl's inference doesn't enforce auth. Network ACLs on the
# cluster ports are the real gate.
export OPENAI_API_KEY_INFERENCE="$INFERENCE_SERVER_API_KEY"
export NCCL_SOCKET_IFNAME

# Build the comma-separated list of base URLs from the inference nodes array.
# prime-rl's client.base-url accepts multiple URLs (round-robin / replica pool).
BASE_URLS=()
for n in "${INFERENCE_NODES[@]}"; do
    BASE_URLS+=("http://${n}:${INFERENCE_PORT}/v1")
done
# Bash array -> CSV
BASE_URL_CSV=$(IFS=','; echo "${BASE_URLS[*]}")

mkdir -p "$OUTPUT_DIR"

# Config is env-overridable for smoke / production swap.
CONFIG=${CONFIG:-configs/rl/qwen35-4b-glyphbench/rl.toml}

echo "[$(hostname)] starting orchestrator (patched compute_advantages)"
echo "  config: $CONFIG"
echo "  inference base-urls: $BASE_URL_CSV"
echo "  output dir: $OUTPUT_DIR"

# We invoke our patch module; it monkey-patches and then runs prime-rl's
# orchestrator main with the same CLI args.
uv run --extra rl --extra eval python -m glyphbench.rl.orchestrator_patch \
    @ "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --client.base-url "$BASE_URL_CSV" \
    --client.api-key-var OPENAI_API_KEY_INFERENCE \
    --weight-broadcast.host "$TRAINER_NODE_IP" \
    --weight-broadcast.port "$NCCL_PORT"
