# `glyphbench.rl`

Custom RL training hooks for prime-rl on the glyphbench benchmark.

## Modules

- `welford.py` — streaming per-key mean/std estimator with a min-σ clamp.
- `advantage.py` — rollout-level GRPO group baseline + per-env σ normalization.
  Drop-in replacement for `prime_rl.orchestrator.advantage.compute_advantages`,
  installed at orchestrator start time via `orchestrator_patch.py`.
- `loss.py` — sequence-mean DPPO+KL loss (each sample contributes equally
  regardless of token count). Wired via prime-rl's documented
  `[trainer.loss] type="custom" import_path` hook.
- `orchestrator_patch.py` — entrypoint that monkey-patches
  `prime_rl.orchestrator.advantage.compute_advantages` to our env-aware
  version, then runs prime-rl's standard orchestrator main.

## Why a monkey-patch?

prime-rl's documented `[advantage] type="custom"` hook receives only
``rewards`` and ``completion_lengths`` — no per-rollout ``env_name``.
Per-env σ tracking needs ``env_name``. The patch replaces one function and
leaves everything else (filters, weight broadcast, eval, checkpoints)
untouched.

## Running

See `scripts/rl/README.md`.
