import pytest

from rl_world_ascii.runner.budget import BudgetExceeded, CostTracker


def test_no_cap_never_exceeds():
    tracker = CostTracker(budget_usd=None)
    tracker.add(1_000_000.0)
    assert tracker.total == 1_000_000.0
    assert not tracker.would_exceed(1.0)


def test_within_cap():
    tracker = CostTracker(budget_usd=10.0)
    tracker.add(3.0)
    tracker.add(4.0)
    assert tracker.total == 7.0
    assert not tracker.would_exceed(2.99)


def test_would_exceed_returns_true_when_projected_over_cap():
    tracker = CostTracker(budget_usd=10.0)
    tracker.add(7.0)
    assert tracker.would_exceed(3.01)


def test_add_raises_budget_exceeded_when_actual_over_cap():
    tracker = CostTracker(budget_usd=5.0)
    tracker.add(4.0)
    with pytest.raises(BudgetExceeded):
        tracker.add(2.0)


def test_none_cost_from_missing_pricing_is_counted_as_zero():
    tracker = CostTracker(budget_usd=10.0)
    tracker.add(None)
    assert tracker.total == 0.0
