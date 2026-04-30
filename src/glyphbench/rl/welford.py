"""Per-key streaming Welford mean/std estimator.

Used by the custom advantage function to track running per-env reward
statistics for σ-normalized GRPO advantages. Numerically stable for the
update sequences we actually run (n < 1e6).

Design choices:
- Per-key dict; we don't enforce a closed key set so that adding a new env
  doesn't require schema migration.
- ``std_clamped`` is what advantage code calls — it returns ``sigma_min``
  for unseen keys or single-observation keys to avoid divide-by-zero.
- Sample std (ddof=1) since we want to estimate the population std from a
  finite sample.
- ``to_dict`` / ``from_dict`` for orchestrator checkpoint integration
  (kept opt-in; v1 doesn't actually checkpoint state — see plan task 18).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class _State:
    """Welford state for a single key."""

    n: int = 0
    mean: float = 0.0
    M2: float = 0.0  # sum of squares of deviations from running mean

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


@dataclass
class PerKeyWelford:
    """Streaming mean/std per string key, with a min-σ clamp."""

    sigma_min: float = 0.1
    _states: dict[str, _State] = field(default_factory=dict)

    def update_batch(self, key: str, xs: Iterable[float]) -> None:
        s = self._states.setdefault(key, _State())
        for x in xs:
            s.update(float(x))

    def mean(self, key: str) -> float:
        if key not in self._states:
            return 0.0
        return self._states[key].mean

    def std_clamped(self, key: str) -> float:
        """σ for key, floored at ``sigma_min`` (also used for unseen keys
        and keys with only one observation)."""
        if key not in self._states:
            return self.sigma_min
        return max(self._states[key].std, self.sigma_min)

    def n(self, key: str) -> int:
        if key not in self._states:
            return 0
        return self._states[key].n

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {k: {"n": s.n, "mean": s.mean, "M2": s.M2} for k, s in self._states.items()}

    @classmethod
    def from_dict(
        cls, d: dict[str, dict[str, float]], sigma_min: float = 0.1
    ) -> "PerKeyWelford":
        instance = cls(sigma_min=sigma_min)
        for k, v in d.items():
            instance._states[k] = _State(n=int(v["n"]), mean=float(v["mean"]), M2=float(v["M2"]))
        return instance
