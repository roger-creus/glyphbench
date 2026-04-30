"""Tests for the per-key Welford running mean/std estimator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from glyphbench.rl.welford import PerKeyWelford


class TestSingleKey:
    def test_first_observation_returns_clamped_sigma(self) -> None:
        w = PerKeyWelford(sigma_min=0.1)
        w.update_batch("env-a", [1.0])
        # n=1 -> variance is 0, std would be 0; clamp engages.
        assert w.std_clamped("env-a") == pytest.approx(0.1)

    def test_unseen_key_returns_clamped_sigma(self) -> None:
        w = PerKeyWelford(sigma_min=0.1)
        assert w.std_clamped("never-seen") == pytest.approx(0.1)

    def test_running_mean_matches_numpy(self) -> None:
        w = PerKeyWelford(sigma_min=1e-9)
        rng = np.random.default_rng(42)
        xs = rng.normal(loc=3.0, scale=2.0, size=500).tolist()
        # Feed in three batches.
        w.update_batch("env-a", xs[:200])
        w.update_batch("env-a", xs[200:350])
        w.update_batch("env-a", xs[350:])
        assert w.mean("env-a") == pytest.approx(float(np.mean(xs)), rel=1e-9)

    def test_running_std_matches_numpy_sample_std(self) -> None:
        w = PerKeyWelford(sigma_min=1e-9)
        rng = np.random.default_rng(7)
        xs = rng.normal(loc=0.0, scale=5.0, size=1000).tolist()
        w.update_batch("env-a", xs)
        # numpy std with ddof=1 is the sample std; Welford produces sample std.
        np_std = float(np.std(xs, ddof=1))
        assert w.std_clamped("env-a") == pytest.approx(np_std, rel=1e-9)


class TestMultipleKeys:
    def test_keys_do_not_bleed(self) -> None:
        w = PerKeyWelford(sigma_min=1e-9)
        w.update_batch("a", [10.0, 12.0, 11.0])
        w.update_batch("b", [-1.0, -2.0, -3.0])
        assert w.mean("a") == pytest.approx(11.0)
        assert w.mean("b") == pytest.approx(-2.0)

    def test_sigma_min_floor_engages_per_key(self) -> None:
        w = PerKeyWelford(sigma_min=0.5)
        # Both keys have only one observation each — std is 0, clamp engages.
        w.update_batch("a", [1.0])
        w.update_batch("b", [100.0])
        assert w.std_clamped("a") == pytest.approx(0.5)
        assert w.std_clamped("b") == pytest.approx(0.5)


class TestSerialization:
    def test_to_dict_from_dict_roundtrip(self) -> None:
        w = PerKeyWelford(sigma_min=0.1)
        w.update_batch("a", [1.0, 2.0, 3.0])
        w.update_batch("b", [10.0, 20.0])
        d = w.to_dict()
        w2 = PerKeyWelford.from_dict(d, sigma_min=0.1)
        assert w2.mean("a") == w.mean("a")
        assert math.isclose(w2.std_clamped("a"), w.std_clamped("a"))
        assert w2.mean("b") == w.mean("b")
