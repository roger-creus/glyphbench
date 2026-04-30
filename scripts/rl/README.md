# `scripts/rl/` — operator guide for glyphbench RL training

Multi-node launch for `prime-rl` with the glyphbench advantage and loss
hooks. Designed for 3× 8 H100 nodes, 1 trainer + 2 inference, no shared FS.

## Prerequisites

1. `git clone … glyphbench` cloned and checked out on **every** node at the
   same path (we don't share an FS, so each node has its own copy).
2. `uv sync --extra rl --extra eval` run once per node.
3. `.env.cluster` filled in at the repo root (see
   `.env.cluster.template`).

## Single-command launch

From the trainer node (`sf-node-1`):

```bash
bash scripts/rl/launch_all.sh
```

This:

1. Runs `health_check.sh --skip-vllm` (SSH + GPUs).
2. SSHes into each inference node and starts vLLM in a `tmux` session
   named `vllm-<hostname>`.
3. Waits up to 10 min for the vLLM endpoints to respond.
4. Starts the patched orchestrator (`tmux` session: `orch`) and the
   trainer (`tmux` session: `trainer`) on the trainer node.

Live logs:

```bash
# vLLM on each inference node
ssh sf-node-2 'tmux attach -t vllm-sf-node-2'
ssh sf-node-3 'tmux attach -t vllm-sf-node-3'

# orchestrator and trainer (on the trainer node)
tmux attach -t orch
tmux attach -t trainer
```

## Manual launch (if `launch_all.sh` doesn't fit your setup)

Each component has its own script:

| Component | Script | Where it runs |
|---|---|---|
| vLLM | `launch_inference.sh` | each inference node |
| orchestrator | `launch_orchestrator.sh` | trainer node |
| trainer | `launch_trainer.sh` | trainer node |

Start them in that order. Inference must be reachable before the
orchestrator boots.

## Health check

```bash
# After your nodes are ready but before launching:
bash scripts/rl/health_check.sh --skip-vllm

# After vLLM is up (e.g. for re-checks during a run):
bash scripts/rl/health_check.sh
```

Writes `$OUTPUT_DIR/control/health.ok` on success.

## Stopping a run

The orchestrator listens for `$OUTPUT_DIR/control/evicted.txt`. Touch it
to make the run exit cleanly:

```bash
echo "manual stop" > "$OUTPUT_DIR/control/evicted.txt"
```

Or kill the tmux sessions:

```bash
tmux kill-session -t trainer
tmux kill-session -t orch
for h in "${INFERENCE_NODES[@]}"; do ssh "$h" "tmux kill-session -t vllm-$h"; done
```

## Troubleshooting

- **`vLLM didn't come up within 10 min`** — SSH into the offending node and
  read `outputs/inference-logs/<host>.log`. Common: HF download failed
  (set `HF_TOKEN`); GPU OOM (lower `--gpu-memory-utilization`).
- **`NCCL connection failed`** — check that `$TRAINER_NODE_IP:$NCCL_PORT`
  is reachable from each inference node. `nc -zv $TRAINER_NODE_IP $NCCL_PORT`
  from an inference node should connect.
- **`Run evicted by trainer`** — orchestrator received a signal in
  `control/evicted.txt`. Check that file's contents for the reason.
