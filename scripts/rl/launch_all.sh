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

# Path on inference nodes where the code lives. May differ from $REPO_ROOT
# on the trainer node (e.g. when using a worktree on sf-node-1 but the main
# clone on inference nodes). Falls back to $REPO_ROOT when paths match.
REMOTE_REPO_ROOT=${REMOTE_REPO_ROOT:-$REPO_ROOT}

# Config + model are env-overridable for smoke / production swap. Passed
# through SSH to the inference nodes' launch_inference.sh.
CONFIG=${CONFIG:-configs/rl/qwen35-4b-glyphbench/rl.toml}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
export CONFIG MODEL MAX_MODEL_LEN

# 1. Pre-flight.
echo "==> health check"
bash scripts/rl/health_check.sh --skip-vllm

# 2. Start vLLM on each inference node, propagating MODEL + MAX_MODEL_LEN.
# Notes:
#   - We pre-create outputs/inference-logs/ on the remote BEFORE tee opens
#     its file (otherwise tee fails silently and the tmux session exits).
#   - We pass MODEL + MAX_MODEL_LEN via tmux's -e flag so the inner shell
#     (which is what runs launch_inference.sh) has them in env. tmux 3.0+
#     supports `-e KEY=VALUE`.
for h in "${INFERENCE_NODES[@]}"; do
    echo "==> starting vLLM on $h (model=$MODEL, repo=$REMOTE_REPO_ROOT)"
    ssh -o BatchMode=yes "$h" "
        export PATH=\"\$HOME/.local/bin:\$PATH\"
        cd '$REMOTE_REPO_ROOT' || { echo 'repo not found at $REMOTE_REPO_ROOT on $h' >&2; exit 1; }
        mkdir -p outputs/inference-logs
        tmux kill-session -t vllm-$h 2>/dev/null || true
        tmux new-session -d -s vllm-$h \
            -e MODEL='$MODEL' \
            -e MAX_MODEL_LEN='$MAX_MODEL_LEN' \
            'bash scripts/rl/launch_inference.sh 2>&1 | tee outputs/inference-logs/$h.log'
        echo \"[$h] tmux new-session exit=\$?\"
        tmux ls 2>&1 | grep vllm-$h || echo \"[$h] WARN: vllm-$h tmux session not visible\"
    "
done

# Resolve SSH-config aliases (sf-node-*) to IPs for curl/HTTP — system
# DNS doesn't know these hostnames; only ~/.ssh/config does.
ssh_resolved_host() {
    ssh -G "$1" 2>/dev/null | awk '/^hostname / {print $2}'
}

# Build {hostname -> ip} for the inference nodes for use in curl/HTTP probes.
declare -A IFER_IP
for h in "${INFERENCE_NODES[@]}"; do
    IFER_IP[$h]=$(ssh_resolved_host "$h")
    echo "  resolved $h -> ${IFER_IP[$h]}"
done

# 3. Wait for vLLM HTTP endpoints to come up (up to 15 min — first model
# load is slow). No auth header (prime-rl's inference doesn't enforce auth).
echo "==> waiting for vllm endpoints (up to 15 min for model load)"
deadline=$((SECONDS + 900))
while [ $SECONDS -lt $deadline ]; do
    all_up=1
    for h in "${INFERENCE_NODES[@]}"; do
        ip="${IFER_IP[$h]}"
        if ! curl -fsS --max-time 4 "http://${ip}:${INFERENCE_PORT}/v1/models" >/dev/null 2>&1; then
            all_up=0
            break
        fi
    done
    if [ "$all_up" = "1" ]; then
        echo "    all inference nodes up"
        break
    fi
    sleep 15
done
if [ "$all_up" != "1" ]; then
    echo "FAIL: vLLM didn't come up within 15 min. Check tmux sessions on inference nodes." >&2
    echo "  ssh sf-node-2 'tmux attach -t vllm-sf-node-2'  # to see live log"
    exit 1
fi

# 4. Start orchestrator + trainer locally (CONFIG passed through env).
echo "==> starting orchestrator (tmux: orch, config=$CONFIG)"
tmux kill-session -t orch 2>/dev/null || true
tmux new-session -d -s orch "CONFIG='$CONFIG' bash scripts/rl/launch_orchestrator.sh 2>&1 | tee $OUTPUT_DIR/orchestrator.log"

echo "==> starting trainer (tmux: trainer, config=$CONFIG)"
tmux kill-session -t trainer 2>/dev/null || true
tmux new-session -d -s trainer "CONFIG='$CONFIG' bash scripts/rl/launch_trainer.sh 2>&1 | tee $OUTPUT_DIR/trainer.log"

echo
echo "All components launched. Re-attach with:"
for h in "${INFERENCE_NODES[@]}"; do
    echo "  ssh $h 'tmux attach -t vllm-$h'"
done
echo "  tmux attach -t orch"
echo "  tmux attach -t trainer"
