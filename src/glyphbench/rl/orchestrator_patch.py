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


def main() -> None:
    # Apply the patch BEFORE importing the orchestrator entrypoint so its
    # references resolve to our function.
    import prime_rl.orchestrator.advantage as _adv

    from glyphbench.rl.advantage import compute_advantages_with_env_norm

    _adv.compute_advantages = compute_advantages_with_env_norm

    # The orchestrator entrypoint reads ``[advantage]`` from the config,
    # but our patch bypasses the registered hook entirely. Set the
    # config's advantage to ``type="default"`` (or just leave it default)
    # so the validators don't complain — our patched function ignores it.

    # Now run the orchestrator.
    from prime_rl.entrypoints.orchestrator import main as orch_main

    orch_main()


if __name__ == "__main__":
    main()
