"""Entrypoint that monkey-patches prime-rl's compute_advantages and runs
the standard prime-rl orchestrator main.

Why: prime-rl's documented ``[advantage] type="custom"`` hook receives only
``rewards`` and ``completion_lengths`` — no ``env_name`` per rollout. We
need ``env_name`` for per-env σ tracking, so we replace one function on the
prime-rl module before its orchestrator imports it.

Run:

    uv run python -m glyphbench.rl.orchestrator_patch @ <orch.toml>
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def _validate_seq_len_matches_max_model_len() -> None:
    """Fail fast if seq_len ≠ max_model_len in any toml passed via argv.

    prime-rl's ``trainer/rl/packer.py`` (search ``> self.seq_len``) does NOT
    silently truncate samples whose packed length exceeds ``seq_len`` — it
    evicts the entire run with "Run wrote a sample with invalid data".
    With one run that terminates training. So if vLLM's ``max_model_len``
    is larger than the trainer's ``seq_len``, vLLM may emit a completion
    that kills the trainer the first time a long rollout happens.

    The two values must be kept equal. This check parses every toml path
    in argv (pyrallis ``@ path/to/file.toml`` syntax) and asserts that any
    file declaring BOTH ``[orchestrator] seq_len`` (or top-level
    ``seq_len``) AND ``[inference.model] max_model_len`` agrees on a
    single value. Split-toml workflows where the two live in different
    files get a soft warning instead of a hard fail (we can't cross-check
    what we can't see).
    """
    candidate_paths: list[Path] = []
    for arg in sys.argv[1:]:
        # pyrallis accepts ``@ file.toml`` (two argv tokens) and
        # ``@file.toml`` (one token). Handle both.
        if arg == "@":
            continue
        if arg.startswith("@"):
            arg = arg[1:]
        if arg.endswith(".toml"):
            p = Path(arg)
            if p.exists():
                candidate_paths.append(p)

    seq_lens: dict[Path, int] = {}
    max_model_lens: dict[Path, int] = {}
    for p in candidate_paths:
        try:
            cfg = tomllib.loads(p.read_text())
        except Exception:
            continue
        # seq_len lives at top level OR under [orchestrator].
        sl = cfg.get("seq_len")
        if sl is None:
            sl = cfg.get("orchestrator", {}).get("seq_len")
        if sl is not None:
            seq_lens[p] = int(sl)
        mml = cfg.get("inference", {}).get("model", {}).get("max_model_len")
        if mml is not None:
            max_model_lens[p] = int(mml)

    # Same-file equality check — strongest signal, hard fail.
    for p, sl in seq_lens.items():
        mml = max_model_lens.get(p)
        if mml is not None and sl != mml:
            raise RuntimeError(
                f"Config error in {p}: trainer seq_len={sl} must equal "
                f"inference max_model_len={mml}. prime-rl evicts the entire "
                f"run if any sample exceeds seq_len (no silent truncation), "
                f"so vLLM cannot be allowed to emit completions longer than "
                f"what the trainer can pack. Set both keys to the same value."
            )

    # Cross-file equality check — both sides seen but in different tomls.
    if seq_lens and max_model_lens:
        sl_vals = set(seq_lens.values())
        mml_vals = set(max_model_lens.values())
        if not sl_vals & mml_vals:
            raise RuntimeError(
                f"Config error across split tomls: trainer seq_len ∈ {sl_vals} "
                f"does not intersect inference max_model_len ∈ {mml_vals}. "
                f"They must be equal — see the same-file message above for why."
            )


def main() -> None:
    # Step 0: cheap config sanity check before we spin anything up.
    _validate_seq_len_matches_max_model_len()

    # Step 1: import the advantage module (cached in sys.modules).
    import prime_rl.orchestrator.advantage as _adv

    from glyphbench.rl.advantage import compute_advantages_with_env_norm

    # Step 2: replace the canonical reference. Any subsequent
    # ``from prime_rl.orchestrator.advantage import compute_advantages``
    # in another module body will pick up our patched function, because
    # ``from import`` resolves attributes against the cached module object.
    _adv.compute_advantages = compute_advantages_with_env_norm

    # Step 3: import the orchestrator main. Its module body does
    # ``from prime_rl.orchestrator.advantage import compute_advantages``,
    # which now binds to our patched function.
    from prime_rl.orchestrator.orchestrator import main as orch_main

    # Step 4: belt-and-braces — also rebind the orchestrator module's
    # local name in case some other module imported orchestrator.orchestrator
    # (and thus snapshotted the original ``compute_advantages``) before us.
    import prime_rl.orchestrator.orchestrator as _orch

    if _orch.compute_advantages is not compute_advantages_with_env_norm:
        _orch.compute_advantages = compute_advantages_with_env_norm

    orch_main()


if __name__ == "__main__":
    main()
