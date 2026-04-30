# `qwen35-4b-glyphbench` RL training config

Trains `Qwen/Qwen3.5-4B` on all 292 glyphbench envs under the same inference
profile we eval at: thinking on, `max_output_tokens=8192`,
`memory_update_max_tokens=4096`, memory mode, frame-stack 0.

## Files

- `rl.toml` — main prime-rl config. Cluster-agnostic (no IPs / API keys).
- (gitignored) `cluster.local.toml` — your per-cluster overrides if needed
  (e.g. `output_dir`, `client.base_url`). Most settings flow from
  environment variables; this file is only needed for things you want
  hard-pinned to a config.

## Cluster setup

The launch scripts read environment variables from `.env.cluster` at the
repo root (gitignored). Required entries:

```bash
# .env.cluster
TRAINER_NODE=sf-node-1
INFERENCE_NODES=(sf-node-2 sf-node-3)
INFERENCE_PORT=8000
INFERENCE_SERVER_API_KEY=<random-string>
NCCL_PORT=29501
NCCL_SOCKET_IFNAME=ens1
OUTPUT_DIR=/home/roger/glyphbench/outputs/qwen35-4b-glyphbench-v1
WANDB_API_KEY=<your-wandb-key>
```

## Launch

From `sf-node-1` (the trainer node):

```bash
bash scripts/rl/launch_all.sh
```

This SSHes to inference nodes to start vLLM, then starts the orchestrator
+ trainer locally. To launch components manually, see
`scripts/rl/README.md`.

## Sizing reference

- 256 rollouts per step (32 problems × 8 rollouts).
- ~10 min per step at 16K seq, 8K + 4K memory budgets.
- 1000 steps ≈ 7 days end-to-end on 24× H100.

## Pinned-decision summary

See `specs/2026-04-30-glyphbench-rl-training-design.md` for full design
notes. Key choices:

- **Advantage**: rollout-level GRPO + per-env Welford σ (custom function,
  monkey-patched in via `glyphbench.rl.orchestrator_patch`).
- **Loss**: sequence-mean DPPO+KL (`glyphbench.rl.loss.sequence_mean_loss`),
  registered via prime-rl's documented loss-import-path hook.
- **Async**: `max_async_level=1`. NCCL weight broadcast.
- **Compute**: 1 trainer node (sf-node-1, FSDP=8) + 2 inference nodes
  (sf-node-2/3, vLLM DP=8 per node, TP=1).
