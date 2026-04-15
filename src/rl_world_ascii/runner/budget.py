"""Cost tracker with hard-abort budget cap."""

from __future__ import annotations

import threading


class BudgetExceeded(RuntimeError):
    """Raised when the cumulative spend exceeds the configured budget cap."""


class CostTracker:
    def __init__(self, budget_usd: float | None) -> None:
        self._budget = budget_usd
        self._total = 0.0
        self._lock = threading.Lock()

    @property
    def total(self) -> float:
        with self._lock:
            return self._total

    @property
    def budget(self) -> float | None:
        return self._budget

    def would_exceed(self, projected_add: float) -> bool:
        if self._budget is None:
            return False
        with self._lock:
            return (self._total + projected_add) > self._budget

    def add(self, cost: float | None) -> None:
        """Add a known cost to the tracker. None is treated as 0 (missing pricing)."""
        amount = float(cost) if cost is not None else 0.0
        with self._lock:
            self._total += amount
            if self._budget is not None and self._total > self._budget:
                raise BudgetExceeded(
                    f"total cost ${self._total:.4f} exceeded budget ${self._budget:.2f}"
                )
