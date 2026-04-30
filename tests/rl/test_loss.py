"""Tests for the sequence-mean DPPO+KL loss.

The key property we test: when two samples differ only in length but
contribute identical per-token gradient, the long sample does NOT dominate
the loss / gradient.
"""

from __future__ import annotations

import torch

from glyphbench.rl.loss import sequence_mean_loss


def make_inputs(seq_len: int, advantage: float = 1.0) -> dict:
    """Build a minimal LossInputs-like object."""
    # Inference logprobs uniform; trainer logprobs equal (importance ratio = 1).
    inference_logprobs = torch.full((seq_len,), -1.0)
    trainer_logprobs = inference_logprobs.clone()
    advantages = torch.full((seq_len,), float(advantage))
    loss_mask = torch.ones(seq_len, dtype=torch.bool)
    teacher_logprobs = None

    # The real LossInputs is a dataclass with these field names. We mimic it
    # via a lightweight namespace so we can call the loss directly without
    # importing prime-rl in tests.
    class _Inputs:
        pass

    inputs = _Inputs()
    inputs.trainer_logprobs = trainer_logprobs
    inputs.inference_logprobs = inference_logprobs
    inputs.teacher_logprobs = teacher_logprobs
    inputs.advantages = advantages
    inputs.loss_mask = loss_mask
    return inputs


def test_loss_is_finite_and_negative_for_positive_advantage() -> None:
    inputs = make_inputs(seq_len=100, advantage=1.0)
    out = sequence_mean_loss(inputs)
    assert torch.isfinite(out.loss)
    # With ratio=1 and positive advantage, pg term is +1; -pg + small KL = -1.
    assert out.loss.item() < 0.0


def test_long_sequence_does_not_dominate_short() -> None:
    """Two samples, different lengths, identical per-token contribution.
    Their losses should be EQUAL (both equal the per-token mean)."""
    short = make_inputs(seq_len=10, advantage=1.0)
    long = make_inputs(seq_len=1000, advantage=1.0)
    out_short = sequence_mean_loss(short)
    out_long = sequence_mean_loss(long)
    # Sequence-mean — equal regardless of length.
    assert torch.isclose(out_short.loss, out_long.loss, rtol=1e-6)


def test_zero_advantage_produces_only_kl_term() -> None:
    """When ratio=1 AND advantage=0, pg=0 and kl=0, so loss=0."""
    inputs = make_inputs(seq_len=50, advantage=0.0)
    out = sequence_mean_loss(inputs)
    assert out.loss.item() == 0.0


def test_metrics_include_n_trainable() -> None:
    inputs = make_inputs(seq_len=50, advantage=1.0)
    out = sequence_mean_loss(inputs)
    assert "n_trainable" in out.metrics
    assert out.metrics["n_trainable"].item() == 50.0
