"""Custom GRPO + per-env σ advantage function for glyphbench RL training.

Drop-in replacement for ``prime_rl.orchestrator.advantage.compute_advantages``.

Why we don't use prime-rl's documented ``[advantage] type="custom"`` hook:
the documented hook only sees ``rewards`` and ``completion_lengths`` — no
per-rollout ``env_name``. Per-env σ requires ``env_name``. This module is
installed via ``orchestrator_patch.py`` which monkey-patches
``prime_rl.orchestrator.advantage.compute_advantages`` at import time.

What we do per call:
1. Group rollouts by ``(env_name, example_id)`` (the ``rollouts_per_example``
   GRPO groups).
2. For each group, compute the within-group mean reward as the baseline.
3. Update per-env Welford σ with this batch's rewards.
4. Assign ``rollout["advantage"] = (R_i − group_mean) / σ_env_clamped``.
5. Attach per-env welford stats to ``rollout["metrics"]`` so prime-rl's
   per-env metric aggregation logs them automatically.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from glyphbench.rl.welford import PerKeyWelford


@dataclass
class GlyphbenchAdvantageState:
    """Module-level state held across compute_advantages calls.

    The orchestrator runs in a single process, so a module-global instance
    is fine. Held in a dataclass to make tests independent of ordering.
    """

    sigma_min: float = 0.1
    welford: PerKeyWelford = field(default_factory=lambda: PerKeyWelford(sigma_min=0.1))

    def __post_init__(self) -> None:
        # Keep welford's clamp in sync with state's clamp.
        self.welford.sigma_min = self.sigma_min


# Module-level singleton, lazily initialized at first call.
_DEFAULT_STATE: GlyphbenchAdvantageState | None = None


def _get_default_state() -> GlyphbenchAdvantageState:
    global _DEFAULT_STATE
    if _DEFAULT_STATE is None:
        _DEFAULT_STATE = GlyphbenchAdvantageState(sigma_min=0.1)
    return _DEFAULT_STATE


def compute_advantages_with_env_norm(
    rollouts: list[dict[str, Any]],
    samples_per_problem: int,
    advantage_config: Any | None = None,  # ignored; kept for signature compat
    state: GlyphbenchAdvantageState | None = None,
) -> None:
    """Compute per-rollout advantages via GRPO group baseline + per-env σ.

    Mutates ``rollouts`` in place. ``samples_per_problem`` must equal the
    actual group size; we sanity-check this.

    Signature matches ``prime_rl.orchestrator.advantage.compute_advantages``
    so this can be monkey-patched in.
    """
    state = state if state is not None else _get_default_state()

    # 1. Group by (env_name, example_id).
    groups: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    for r in rollouts:
        key = (r["env_name"], r["example_id"])
        groups[key].append(r)

    # Sanity-check group sizes (prime-rl guarantees this; we assert defensively).
    for key, members in groups.items():
        if len(members) != samples_per_problem:
            raise ValueError(
                f"Group {key!r} has {len(members)} rollouts, "
                f"expected samples_per_problem={samples_per_problem}"
            )

    # 2. Compute baselines and update Welford state per env.
    for (env_name, _example_id), members in groups.items():
        rewards = [float(r["reward"]) for r in members]
        baseline = sum(rewards) / len(rewards)

        # Update before we read sigma so this batch's data is in the estimator.
        state.welford.update_batch(env_name, rewards)
        sigma = state.welford.std_clamped(env_name)

        for r in members:
            advantage = (float(r["reward"]) - baseline) / sigma
            r["advantage"] = advantage

            # Attach per-env welford stats to the rollout's metrics dict so
            # prime-rl's per-env aggregation logs them under
            # `metrics/<env_name>/welford_env_*`.
            metrics = r.setdefault("metrics", {})
            metrics["welford_env_mean"] = state.welford.mean(env_name)
            metrics["welford_env_std"] = state.welford.std_clamped(env_name)
            metrics["welford_env_n"] = float(state.welford.n(env_name))


def get_global_advantage_state() -> GlyphbenchAdvantageState:
    """Accessor for inspection / testing — returns the singleton instance."""
    return _get_default_state()
