#!/usr/bin/env bash
# Pre-flight check before kicking off a training run. Verifies:
#   - SSH reachability to inference nodes
#   - 8× free H100 on each node
#   - **All GPUs idle** — no compute processes from any user on any GPU
#     (we won't compete with another tenant). Override with --skip-busy-gpu
#     ONLY if you've personally confirmed the running processes are yours
#     and benign.
#   - NCCL port open from inference nodes back to trainer (implicit via SSH).
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
SKIP_BUSY_GPU=0
for arg in "$@"; do
    case "$arg" in
        --skip-vllm) SKIP_VLLM=1 ;;
        --skip-busy-gpu) SKIP_BUSY_GPU=1 ;;
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

# Verifies NO compute processes are running on ANY GPU on the node — we
# refuse to share with another tenant. The check uses nvidia-smi's
# --query-compute-apps which lists pid, process_name, used_memory for every
# active CUDA context across all GPUs. Any non-empty output = busy.
check_local_gpu_processes() {
    if [ "$SKIP_BUSY_GPU" = "1" ]; then
        echo "  SKIP: GPU-process check disabled by --skip-busy-gpu"
        return
    fi
    local apps
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
    if [ -n "$apps" ]; then
        echo "  FAIL: GPU(s) busy with running compute processes:"
        echo "$apps" | sed 's/^/    /'
        echo "    Refusing to launch alongside another tenant."
        echo "    Override with --skip-busy-gpu ONLY if you've confirmed these are your own processes."
        EXIT=1
    else
        echo "  OK:   no compute processes on any GPU"
    fi
}

check_remote_gpu_processes() {
    if [ "$SKIP_BUSY_GPU" = "1" ]; then
        echo "  SKIP: GPU-process check disabled by --skip-busy-gpu"
        return
    fi
    local host="$1"
    local apps
    apps=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null' 2>/dev/null || echo "ssh-failed")
    if [ "$apps" = "ssh-failed" ]; then
        echo "  FAIL: $host could not query GPU processes (SSH failed)"
        EXIT=1
    elif [ -n "$apps" ]; then
        echo "  FAIL: $host has GPU(s) busy with compute processes:"
        echo "$apps" | sed 's/^/    /'
        EXIT=1
    else
        echo "  OK:   $host: no compute processes on any GPU"
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
check_local_gpu_processes

echo
echo "== Inference nodes =="
for h in "${INFERENCE_NODES[@]}"; do
    echo "-- $h --"
    check_ssh "$h"
    check_remote_gpus "$h"
    check_remote_gpu_processes "$h"
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
