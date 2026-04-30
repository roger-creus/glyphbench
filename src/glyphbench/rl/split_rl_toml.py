"""Split a unified RL toml (rl.toml-style) into per-component sub-tomls.

prime_rl ships a unified `rl.toml` schema (`prime_rl.configs.rl.RLConfig`)
with `[trainer.*]`, `[orchestrator.*]`, `[inference.*]`, plus shared
top-level blocks (`[weight_broadcast]`, `[wandb]`, `[model]`, `[ckpt]`,
`[deployment]`, `[slurm]`). The unified `prime_rl.entrypoints.rl`
splits this into per-component tomls and spawns child processes —
useful when running on one node, but not what the manual launchers in
`scripts/rl/launch_*.sh` do.

The manual launchers invoke the orchestrator (`prime_rl.orchestrator`)
and trainer (`prime_rl.trainer.rl.train`) directly. Those entrypoints
are bound to OrchestratorConfig / TrainerConfig respectively, both of
which `forbid` extra inputs — so passing the unified rl.toml to them
fails with `Extra inputs are not permitted` for every top-level block.

This helper loads RLConfig, lets its model_validators propagate shared
fields (`weight_broadcast` etc.) into both subconfigs, then dumps:

    <out>/orchestrator.toml
    <out>/trainer.toml

Each is a self-contained config the matching prime-rl entrypoint can
consume. Inference is invoked with CLI args (no toml), so we don't
emit `inference.toml`.

Usage:
    python -m glyphbench.rl.split_rl_toml <unified.toml> <out_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import tomli
import tomli_w

from prime_rl.configs.rl import RLConfig


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <unified.toml> <out_dir>")
    src = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)

    with open(src, "rb") as f:
        cfg_dict = tomli.load(f)
    cfg = RLConfig.model_validate(cfg_dict)

    trainer_path = out / "trainer.toml"
    orch_path = out / "orchestrator.toml"
    with open(trainer_path, "wb") as f:
        tomli_w.dump(cfg.trainer.model_dump(exclude_none=True, mode="json"), f)
    with open(orch_path, "wb") as f:
        tomli_w.dump(cfg.orchestrator.model_dump(exclude_none=True, mode="json"), f)
    print(f"trainer={trainer_path}")
    print(f"orchestrator={orch_path}")


if __name__ == "__main__":
    main()
