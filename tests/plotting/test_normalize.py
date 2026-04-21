"""Tests for normalization module."""

import pytest

from glyphbench.plotting.normalize import (
    REFERENCES,
    benchmark_aggregate,
    normalized_score,
)


class TestNormalizedScore:
    def test_known_env(self) -> None:
        score = normalized_score("glyphbench/atari-pong-v0", 14.6)
        assert score is not None
        assert abs(score - 1.0) < 0.01  # expert return = normalized 1.0

    def test_random_return(self) -> None:
        score = normalized_score("glyphbench/atari-pong-v0", -20.7)
        assert score is not None
        assert abs(score) < 0.01  # random return = normalized 0.0

    def test_unknown_env(self) -> None:
        assert normalized_score("glyphbench/nonexistent-v0", 5.0) is None

    def test_above_expert(self) -> None:
        score = normalized_score("glyphbench/atari-freeway-v0", 35.0)
        assert score is not None
        assert score > 1.0  # above expert


class TestBenchmarkAggregate:
    def test_median(self) -> None:
        scores: dict[str, float | None] = {"a": 0.5, "b": 0.8, "c": 0.3}
        assert benchmark_aggregate(scores, "median") == pytest.approx(0.5)

    def test_mean(self) -> None:
        scores: dict[str, float | None] = {"a": 0.3, "b": 0.6, "c": 0.9}
        assert benchmark_aggregate(scores, "mean") == pytest.approx(0.6)

    def test_iqm(self) -> None:
        scores: dict[str, float | None] = {"a": 0.1, "b": 0.4, "c": 0.6, "d": 0.9}
        result = benchmark_aggregate(scores, "iqm")
        assert 0.3 < result < 0.7

    def test_ignores_none(self) -> None:
        scores: dict[str, float | None] = {"a": 0.5, "b": None, "c": 0.7}
        assert benchmark_aggregate(scores, "median") == pytest.approx(0.6)

    def test_empty(self) -> None:
        assert benchmark_aggregate({}) == 0.0


class TestReferences:
    def test_references_not_empty(self) -> None:
        assert len(REFERENCES) > 0

    def test_all_have_random_return(self) -> None:
        for ref in REFERENCES.values():
            assert isinstance(ref.random_return, float)
