# `qwen35-4b-glyphbench` RL training config

Trains `Qwen/Qwen3.5-4B` on the 284 trainable glyphbench envs (atari and
craftaxfull are excluded by policy — long-horizon, archived). Memory mode
on, thinking on, `max_output_tokens=8192`, `memory_update_max_tokens=4096`,
frame-stack 0. Same inference profile as the offline eval CLI.

## Files

- `rl.toml` — main prime-rl config. Cluster-agnostic (no IPs, no API keys,
  no hostnames). Bring your own launcher.

## Hardware assumptions

The defaults target **1 trainer node × 8 GPUs (H100 / 80 GB) + 1 inference
node × 8 GPUs**. Adjust `[deployment]`, `[inference.parallel]`, and
`fused_lm_head_token_chunk_size` for different fleets.

- Trainer: FSDP across 8 GPUs, full activation checkpointing, fused chunked
  LM head (vanilla LM head OOMs on Qwen3.5-4B's 152K vocab × 24K seq).
- Inference: vLLM, `tp=1, dp=8`, `max_model_len=24576` matched to trainer
  `seq_len=24576` (action calls get up to 16384-token prompts, memory
  calls up to 20480). The two **must** be equal: prime-rl's packer evicts
  the entire run if any sample exceeds `seq_len` — there is no silent
  truncation, so an oversized vLLM completion kills training.

If your two nodes can open arbitrary TCP ports to each other, switch
`[weight_broadcast] type = "nccl"` for faster broadcasts. Filesystem
broadcast (the default here) requires a shared path between trainer and
inference nodes (NFS, sshfs, etc.).

## Sizing reference

- **32 rollouts/step** = `batch_size = 32` (prime-rl semantic: total
  rollouts, not unique examples) ÷ `rollouts_per_example = 8` (GRPO group
  size) → **4 unique tasks/step × 8 rollouts each**.
- `seq_len = 24576` trainer-side, matching `max_model_len = 24576` on the
  inference side. No rollout segments can exceed seq_len.
- Eval cycles (every 25 steps) = 240 rollouts across 30 task slots; add
  ~15-90 min depending on env wall-time. At 32 train rollouts/step, eval
  cycles are noticeable overhead — adjust `[orchestrator.eval] interval`
  upward if eval-vs-train compute ratio matters.
- 1000 steps wall-clock: highly dependent on per-rollout completion length;
  expect on the lower end of 4-12 h with this small batch.

## Pinned-decision summary

- **Advantage**: rollout-level GRPO + per-env Welford σ.
  - `[orchestrator.advantage] type = "default"` is a placeholder that exists
    only so prime-rl's pydantic validator passes. The actual advantage
    function is installed via a monkey-patch at orchestrator import time.
  - Launch the orchestrator via:
    `uv run python -m glyphbench.rl.orchestrator_patch @ orchestrator.toml`
    (NOT the stock prime-rl orchestrator entrypoint).
  - Source: `glyphbench/rl/{advantage.py, welford.py, orchestrator_patch.py}`.
- **Loss**: sequence-mean DPPO+KL — `glyphbench.rl.loss.sequence_mean_loss`,
  registered via prime-rl's documented loss-import-path hook.
- **Async**: `max_async_level = 1`.
- **Memory mode**: each env step is two LLM calls (action + memory update),
  stored as one trainable trajectory segment.
- **Reward**: `episodic_return` only (weight=1.0). All format / parse-rate
  signals (`forfeit_rate`, `xml_format_reward`, `*_truncation_rate`) are
  registered as weight=0 metrics — logged but never enter the gradient.

## Eval picks

`eval_interval = 25` grad steps. 30 slots = 5 tasks × 6 surviving training
suites, one block per task_id so the per-task summary line is visible in
the orchestrator log:

| suite | task_ids (5 each) |
|---|---|
| classics  | sokoban-easy, lightsout-hard, maze-medium, icesliding-easy, nonogram-easy |
| craftax   | fight-bats, wave-defense, firstday, craft-ironset, fight-skeletons |
| miniatari | defender, pong, skiing, asteroids, bankheist |
| minigrid  | keycorridor-s5r3, obstructedmaze-1dlh, dynamic-obstacles-16x16, lockedroom, unlockpickup |
| minihack  | hidenseek-big, hidenseek-lava, mazewalk-9x9, river-lava, room-ultimate-5x5 |
| procgen   | starpilot, caveflyer, heist, jumper, fruitbot |

Each runs `num_examples=4 × rollouts_per_example=2 = 8 rollouts`. Total per
eval cycle: 30 × 8 = 240 rollouts. Tune for your infra by editing or
pruning `[[orchestrator.eval.env]]` blocks. The full 292-env sweep is the
offline `glyphbench eval` CLI, not in-loop eval.
