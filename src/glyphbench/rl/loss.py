"""Sequence-mean DPPO+KL loss for glyphbench RL training.

Same DPPO+KL ingredients as ``prime_rl.trainer.rl.loss.default_loss_fn``
(importance-ratio masking + KL-mismatch penalty), but the per-sample loss is
normalized by THAT SAMPLE's trainable token count rather than left as a sum
over tokens. After prime-rl's outer ``compute_loss`` accumulates over
samples and divides by ``loss_scale`` (total trainable tokens), each sample
contributes its per-token mean — every rollout-sample weights equally
regardless of length.

Note on absolute scale: prime-rl's ``loss_scale`` is total trainable tokens.
Our per-sample mean changes the denominator's MEANING but not the
length-fairness property. Effective LR will differ vs. the default loss by
~(avg trainable tokens per sample) — typically O(1e3) for our setup. Tune
LR after the first smoke step's gradient norm logs.

Wire-in via:

    [trainer.loss]
    type = "custom"
    import_path = "glyphbench.rl.loss.sequence_mean_loss"
    kwargs = { dppo_mask_low = 0.2, dppo_mask_high = 0.2, adv_tau = 1.0, kl_tau = 1e-3 }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class _SeqMeanOutputs:
    """Mimics ``prime_rl.trainer.rl.loss.LossOutputs`` to avoid a
    runtime import of prime-rl's typed dataclass at the test boundary."""

    loss: Tensor
    metrics: dict[str, Tensor] = field(default_factory=dict)


def _safe_mean(values: Tensor, mask: Tensor) -> Tensor:
    denom = torch.clamp_min(mask.sum(), 1)
    return values[mask].sum() / denom


def sequence_mean_loss(
    inputs: Any,
    dppo_mask_low: float = 0.2,
    dppo_mask_high: float = 0.2,
    adv_tau: float = 1.0,
    kl_tau: float = 1e-3,
) -> Any:
    """Drop-in replacement for ``default_loss_fn`` with sequence-mean
    normalization.

    ``inputs`` is the prime-rl ``LossInputs`` dataclass (or any object with
    ``trainer_logprobs``, ``inference_logprobs``, ``advantages``,
    ``loss_mask`` attributes — see prime-rl's
    ``docs/bring-your-own-algorithms.md``).
    """
    trainer_logprobs: Tensor = inputs.trainer_logprobs
    inference_logprobs: Tensor = inputs.inference_logprobs
    advantages: Tensor = inputs.advantages
    loss_mask: Tensor = inputs.loss_mask

    # Same DPPO masking machinery as the default loss.
    trainer_probs = torch.exp(trainer_logprobs)
    inference_probs = torch.exp(inference_logprobs)
    probs_diff = trainer_probs - inference_probs
    dppo_invalid_mask_high = probs_diff > dppo_mask_high
    dppo_invalid_mask_low = probs_diff < -dppo_mask_low
    dppo_invalid_mask = torch.where(
        advantages > 0, dppo_invalid_mask_high, dppo_invalid_mask_low
    )

    is_masked = dppo_invalid_mask
    is_masked_high = (advantages > 0) & dppo_invalid_mask_high
    is_masked_low = (advantages < 0) & dppo_invalid_mask_low
    keep_mask = loss_mask & ~is_masked

    log_importance_ratio = trainer_logprobs - inference_logprobs
    importance_ratio = torch.exp(log_importance_ratio)
    mismatch_kl = importance_ratio - log_importance_ratio - 1

    advantages_scaled = adv_tau * advantages

    pg_loss = keep_mask * advantages_scaled * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2

    raw_per_token = -pg_loss + kl_tau * kl_loss  # [seq]

    # Sequence mean: divide by THIS sample's trainable token count.
    n_trainable = loss_mask.sum().clamp_min(1).to(raw_per_token.dtype)
    loss = raw_per_token.sum() / n_trainable

    metrics = {
        "n_trainable": n_trainable,
        "mismatch_kl": _safe_mean(mismatch_kl, loss_mask),
        "is_masked": _safe_mean(is_masked, loss_mask),
        "is_masked_low": _safe_mean(is_masked_low, loss_mask),
        "is_masked_high": _safe_mean(is_masked_high, loss_mask),
    }

    # Return prime-rl's LossOutputs if available (production path), else our
    # plain dataclass (test path).
    try:
        from prime_rl.trainer.rl.loss import LossOutputs  # type: ignore

        return LossOutputs(loss=loss, metrics=metrics)
    except Exception:
        return _SeqMeanOutputs(loss=loss, metrics=metrics)
