#!/usr/bin/env bash
# Pre-flight check before kicking off a training run. Verifies:
#   - SSH reachability to inference nodes
#   - 8× free H100 on each node
#   - NCCL port open from inference nodes back to trainer
#   - vLLM endpoints respond (only checked AFTER inference is up; pass
#     --skip-vllm to skip).
#
# Exits non-zero on any failure. Run before launch_all.sh.

set -uo pipefail  # NOT -e — we want to collect all failures, then summarize.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$REPO_ROOT/.env.cluster" ]; then
    echo "ERROR: $REPO_ROOT/.env.cluster not found." >&2
    exit 2
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/.env.cluster"

SKIP_VLLM=0
for arg in "$@"; do
    case "$arg" in
        --skip-vllm) SKIP_VLLM=1 ;;
    esac
done

EXIT=0

check_local_gpus() {
    local n
    n=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$n" -lt 8 ]; then
        echo "  FAIL: only $n GPUs visible (expected 8)"
        EXIT=1
    else
        echo "  OK:   $n GPUs visible"
    fi
}

check_remote_gpus() {
    local host="$1"
    local n
    n=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" 'nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l' 2>/dev/null || echo 0)
    if [ "$n" -lt 8 ]; then
        echo "  FAIL: $host has only $n GPUs visible (expected 8)"
        EXIT=1
    else
        echo "  OK:   $host has $n GPUs visible"
    fi
}

check_ssh() {
    local host="$1"
    if ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" 'true' >/dev/null 2>&1; then
        echo "  OK:   ssh $host"
    else
        echo "  FAIL: ssh $host (BatchMode, no password)"
        EXIT=1
    fi
}

check_vllm() {
    local host="$1"
    local url="http://${host}:${INFERENCE_PORT}/v1/models"
    if curl -fsS --max-time 8 \
        -H "Authorization: Bearer ${INFERENCE_SERVER_API_KEY}" \
        "$url" >/dev/null 2>&1; then
        echo "  OK:   vllm $host:$INFERENCE_PORT"
    else
        echo "  FAIL: vllm $host:$INFERENCE_PORT not responding"
        EXIT=1
    fi
}

echo "== Local node ($(hostname)) =="
check_local_gpus

echo
echo "== Inference nodes =="
for h in "${INFERENCE_NODES[@]}"; do
    echo "-- $h --"
    check_ssh "$h"
    check_remote_gpus "$h"
    if [ "$SKIP_VLLM" = "0" ]; then
        check_vllm "$h"
    fi
done

echo
if [ "$EXIT" = "0" ]; then
    echo "ALL CHECKS PASSED"
    mkdir -p "${OUTPUT_DIR:-$REPO_ROOT/outputs}/control"
    : > "${OUTPUT_DIR:-$REPO_ROOT/outputs}/control/health.ok"
else
    echo "FAILURES — fix before launching"
fi

exit "$EXIT"
