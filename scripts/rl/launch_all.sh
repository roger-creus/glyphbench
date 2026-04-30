#!/usr/bin/env bash
# Single-command launch from the trainer node:
#   1. Run health checks.
#   2. SSH into each inference node and start vLLM in a backgrounded
#      tmux session (so we can re-attach for logs).
#   3. Wait for vLLM endpoints to come up.
#   4. Start orchestrator + trainer locally, also in tmux sessions.
#
# Tmux session names: vllm-<host>, orch, trainer.
# Re-attach with: ssh <host> 'tmux attach -t vllm-<host>'
#                tmux attach -t orch
#                tmux attach -t trainer

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$REPO_ROOT/.env.cluster" ]; then
    echo "ERROR: $REPO_ROOT/.env.cluster not found." >&2
    exit 2
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/.env.cluster"

# 1. Pre-flight (skip vLLM check; we're starting it).
echo "==> health check"
bash scripts/rl/health_check.sh --skip-vllm

# 2. Start vLLM on each inference node.
for h in "${INFERENCE_NODES[@]}"; do
    echo "==> starting vLLM on $h"
    ssh -o BatchMode=yes "$h" "
        cd '$REPO_ROOT' || { echo 'repo not found at $REPO_ROOT on $h' >&2; exit 1; }
        tmux kill-session -t vllm-$h 2>/dev/null || true
        tmux new-session -d -s vllm-$h 'bash scripts/rl/launch_inference.sh 2>&1 | tee outputs/inference-logs/$h.log'
    "
done

# 3. Wait for vLLM HTTP endpoints to come up (up to 10 min).
echo "==> waiting for vllm endpoints"
deadline=$((SECONDS + 600))
while [ $SECONDS -lt $deadline ]; do
    all_up=1
    for h in "${INFERENCE_NODES[@]}"; do
        if ! curl -fsS --max-time 4 \
            -H "Authorization: Bearer ${INFERENCE_SERVER_API_KEY}" \
            "http://${h}:${INFERENCE_PORT}/v1/models" >/dev/null 2>&1; then
            all_up=0
            break
        fi
    done
    if [ "$all_up" = "1" ]; then
        echo "    all inference nodes up"
        break
    fi
    sleep 10
done
if [ "$all_up" != "1" ]; then
    echo "FAIL: vLLM didn't come up within 10 min. Check tmux sessions on inference nodes." >&2
    exit 1
fi

# 4. Start orchestrator + trainer locally.
echo "==> starting orchestrator (tmux: orch)"
tmux kill-session -t orch 2>/dev/null || true
tmux new-session -d -s orch "bash scripts/rl/launch_orchestrator.sh 2>&1 | tee $OUTPUT_DIR/orchestrator.log"

echo "==> starting trainer (tmux: trainer)"
tmux kill-session -t trainer 2>/dev/null || true
tmux new-session -d -s trainer "bash scripts/rl/launch_trainer.sh 2>&1 | tee $OUTPUT_DIR/trainer.log"

echo
echo "All components launched. Re-attach with:"
for h in "${INFERENCE_NODES[@]}"; do
    echo "  ssh $h 'tmux attach -t vllm-$h'"
done
echo "  tmux attach -t orch"
echo "  tmux attach -t trainer"
