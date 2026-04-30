#!/usr/bin/env bash
# Start the prime-rl orchestrator on the trainer node, with our
# compute_advantages monkey-patch installed.

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

CUDA_HOME=${CUDA_HOME:-$HOME/cuda-12.6}
if [ -x "$CUDA_HOME/bin/nvcc" ]; then
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
fi

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
: "${INFERENCE_SERVER_API_KEY:=}"
INFERENCE_HTTP_HOST_OVERRIDE=${INFERENCE_HTTP_HOST_OVERRIDE:-}

# Set OPENAI_API_KEY_INFERENCE (used by prime-rl's client.api-key-var) — empty
# is OK because prime-rl's inference doesn't enforce auth. Network ACLs on the
# cluster ports are the real gate.
export OPENAI_API_KEY_INFERENCE="$INFERENCE_SERVER_API_KEY"
export NCCL_SOCKET_IFNAME

# Build the comma-separated list of base URLs from the inference nodes array.
# prime-rl's client.base-url accepts multiple URLs (round-robin / replica pool).
# Resolve SSH-config aliases to IPs because the orchestrator's HTTP client
# uses system DNS (which doesn't know sf-node-*).
BASE_URLS=()
for n in "${INFERENCE_NODES[@]}"; do
    if [ -n "$INFERENCE_HTTP_HOST_OVERRIDE" ]; then
        # SSH-tunnel mode: reach all inference nodes via a single localhost port.
        # (Multi-node tunneling would need per-node distinct local ports.)
        ip="$INFERENCE_HTTP_HOST_OVERRIDE"
    else
        ip=$(ssh -G "$n" 2>/dev/null | awk '/^hostname / {print $2}')
    fi
    BASE_URLS+=("http://${ip}:${INFERENCE_PORT}/v1")
done
BASE_URL_CSV=$(IFS=','; echo "${BASE_URLS[*]}")

mkdir -p "$OUTPUT_DIR"

# Config is env-overridable for smoke / production swap.
CONFIG=${CONFIG:-configs/rl/qwen35-4b-glyphbench/rl.toml}

# prime-rl's orchestrator entrypoint expects an OrchestratorConfig (no
# [trainer]/[inference]/[deployment] siblings), but our shared CONFIG is
# a unified RLConfig. Split it into per-component tomls in a temp dir.
SPLIT_DIR=$(mktemp -d -t rl-split.XXXXXX)
trap 'rm -rf "$SPLIT_DIR"' EXIT
uv run --extra rl --extra eval python -m glyphbench.rl.split_rl_toml "$CONFIG" "$SPLIT_DIR" >/dev/null
ORCH_CONFIG="$SPLIT_DIR/orchestrator.toml"

# prime_rl's MultiRunManager expects orchestrator.output_dir to be
# `<trainer.output_dir>/<run_id>` (validated by validate_shared_output_dir).
# Use the same default ("run_default") that prime_rl.entrypoints.rl sets.
ORCH_OUTPUT_DIR="$OUTPUT_DIR/run_default"
mkdir -p "$ORCH_OUTPUT_DIR"

echo "[$(hostname)] starting orchestrator (patched compute_advantages)"
echo "  unified config: $CONFIG"
echo "  orchestrator subconfig: $ORCH_CONFIG"
echo "  inference base-urls: $BASE_URL_CSV"
echo "  output dir: $ORCH_OUTPUT_DIR (run_default subdir of trainer's output dir)"

# We invoke our patch module; it monkey-patches and then runs prime-rl's
# orchestrator main with the same CLI args.
uv run --extra rl --extra eval python -m glyphbench.rl.orchestrator_patch \
    @ "$ORCH_CONFIG" \
    --output-dir "$ORCH_OUTPUT_DIR" \
    --client.base-url "$BASE_URL_CSV" \
    --client.api-key-var OPENAI_API_KEY_INFERENCE
