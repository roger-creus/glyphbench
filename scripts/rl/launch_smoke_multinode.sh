#!/usr/bin/env bash
# Multi-node smoke test launcher. Uses 1 trainer FSDP=8 + 8 vLLM workers
# on a single inference node on a SHORT run that exercises every code
# path: rollout generation, custom advantage with per-env Welford σ,
# sequence-mean loss, filesystem weight broadcast, checkpoint save,
# eval-during-training, W&B (offline) metric logging.
#
# SF cluster note: nodes are isolated VMs that only share port 2222
# (SSH) — direct TCP for vLLM's HTTP API and NCCL collectives is not
# possible. We work around this by:
#   1. SSH forward tunnel: sf-node-1's localhost:8000 -> sf-node-2:8000
#      so the orchestrator reaches vLLM through localhost.
#   2. Filesystem weight broadcast (instead of NCCL): trainer writes a
#      HF checkpoint to OUTPUT_DIR/.../broadcast/step_N/, marks STABLE,
#      orchestrator polls and tells vLLM workers to load that path.
#      sshfs mount on sf-node-2 makes that path visible from there.
#
# Run from the trainer node (sf-node-1):
#   bash scripts/rl/launch_smoke_multinode.sh
#
# Tmux sessions: vllm-<host>, orch, trainer. The SSH tunnel is a
# detached `ssh -fN` process; the sshfs mount lives on sf-node-2.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$REPO_ROOT/.env.cluster" ]; then
    echo "ERROR: $REPO_ROOT/.env.cluster not found." >&2
    exit 2
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/.env.cluster"

: "${OUTPUT_DIR:?OUTPUT_DIR not set}"
: "${INFERENCE_PORT:?INFERENCE_PORT not set}"
: "${INFERENCE_NODES:?INFERENCE_NODES not set}"

# Single inference node for this smoke (the wrapper's tunnel + sshfs
# logic below assumes 1 inference node — multi-node tunneling would
# need per-node distinct local ports).
if [ "${#INFERENCE_NODES[@]}" -gt 1 ]; then
    echo "ERROR: smoke wrapper expects exactly 1 inference node, got: ${INFERENCE_NODES[*]}" >&2
    exit 2
fi
INFERENCE_HOST="${INFERENCE_NODES[0]}"

if [ -n "${INFERENCE_HTTP_HOST_OVERRIDE:-}" ]; then
    echo "==> tearing down any stale ssh tunnel on localhost:$INFERENCE_PORT"
    pkill -f "ssh.*-L *$INFERENCE_PORT:localhost:$INFERENCE_PORT.*$INFERENCE_HOST" 2>/dev/null || true

    echo "==> ssh forward tunnel: localhost:$INFERENCE_PORT -> $INFERENCE_HOST:$INFERENCE_PORT"
    ssh -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
        -L "$INFERENCE_PORT:localhost:$INFERENCE_PORT" \
        "$INFERENCE_HOST"
    sleep 1
    if ! ss -tln 2>/dev/null | grep -q ":$INFERENCE_PORT "; then
        echo "ERROR: ssh tunnel did not come up on localhost:$INFERENCE_PORT" >&2
        exit 1
    fi
    echo "    tunnel OK"

    echo "==> sshfs mount on $INFERENCE_HOST:$OUTPUT_DIR <- $(hostname):$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    ssh -o BatchMode=yes "$INFERENCE_HOST" "
        set -e
        if mountpoint -q '$OUTPUT_DIR' 2>/dev/null; then
            fusermount3 -u '$OUTPUT_DIR' 2>/dev/null || fusermount -u '$OUTPUT_DIR' 2>/dev/null || true
        fi
        mkdir -p '$OUTPUT_DIR'
        sshfs -o reconnect,ServerAliveInterval=30,cache=no,Compression=no sf-node-1:'$OUTPUT_DIR' '$OUTPUT_DIR'
        mountpoint -q '$OUTPUT_DIR' || { echo 'ERROR: sshfs mount missing' >&2; exit 1; }
        echo '    sshfs mount OK'
    "
fi

# Use the production model (Qwen3.5-4B) so the smoke truly mirrors the
# eval profile — glyphbench's parser is tuned to Qwen3.5's thinking-token
# format, and using a different model would mask format-related bugs.
export CONFIG=configs/rl/qwen35-4b-glyphbench/smoke-multinode.toml
export MODEL=Qwen/Qwen3.5-4B
export MAX_MODEL_LEN=16384

exec bash "$REPO_ROOT/scripts/rl/launch_all.sh"
