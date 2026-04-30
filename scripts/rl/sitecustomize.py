"""Auto-applied at every Python startup in this venv (Python's standard
`site` mechanism imports `sitecustomize` after configuring the path).

When the env var GLYPHBENCH_PATCH_PRIME_RL is set, this module replaces
prime-rl's compute_advantages with our env-aware version. We use this
hook because prime-rl's `rl` entrypoint spawns the orchestrator as a
subprocess via `subprocess.Popen(["orchestrator", ...])`; that
subprocess can't see monkey-patches applied in the parent process.
Setting the env var BEFORE spawning the subprocess (it's inherited by
default) makes this sitecustomize fire inside the orchestrator process
and apply the patch before prime-rl's orchestrator code runs.

This is opt-in — without GLYPHBENCH_PATCH_PRIME_RL set, this file is a
no-op.
"""

from __future__ import annotations

import os


def _maybe_patch_prime_rl() -> None:
    if not os.environ.get("GLYPHBENCH_PATCH_PRIME_RL"):
        return
    try:
        import prime_rl.orchestrator.advantage as _adv
    except ImportError:
        return
    try:
        from glyphbench.rl.advantage import compute_advantages_with_env_norm
    except ImportError:
        return
    _adv.compute_advantages = compute_advantages_with_env_norm


_maybe_patch_prime_rl()
