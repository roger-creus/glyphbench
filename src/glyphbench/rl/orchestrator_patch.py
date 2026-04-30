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


def main() -> None:
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
