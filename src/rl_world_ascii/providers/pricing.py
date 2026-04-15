"""Pricing loader and cost computation.

The single source of truth is `pricing.yaml` at the repo root (or any path
passed to `Pricing.from_yaml`). Loaded lazily by the runner at startup.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelRates:
    input_per_1m_tokens: float
    output_per_1m_tokens: float
    reasoning_per_1m_tokens: float | None


@dataclass
class Pricing:
    version: str
    providers: dict[str, dict[str, ModelRates]]

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Pricing":
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        version = str(data.get("version", "unknown"))
        providers: dict[str, dict[str, ModelRates]] = {}
        for provider_name, models in data.get("providers", {}).items():
            providers[provider_name] = {}
            for model_id, rates in models.items():
                providers[provider_name][model_id] = ModelRates(
                    input_per_1m_tokens=float(rates["input_per_1m_tokens"]),
                    output_per_1m_tokens=float(rates["output_per_1m_tokens"]),
                    reasoning_per_1m_tokens=(
                        float(rates["reasoning_per_1m_tokens"])
                        if rates.get("reasoning_per_1m_tokens") is not None
                        else None
                    ),
                )
        return cls(version=version, providers=providers)

    def compute_cost(
        self,
        *,
        provider: str,
        model_id: str,
        tokens_in: int,
        tokens_out: int,
        tokens_reasoning: int,
    ) -> float | None:
        """Return dollar cost for this call, or None if no pricing entry matches.

        Looks up an exact match on `(provider, model_id)` first. Falls back to
        a wildcard `"*"` entry for the provider if present (used by vLLM and
        mock, which are free regardless of model).
        """
        models = self.providers.get(provider)
        if models is None:
            warnings.warn(
                f"no pricing entry for provider {provider!r}; cost will be None",
                stacklevel=2,
            )
            return None
        rates = models.get(model_id) or models.get("*")
        if rates is None:
            warnings.warn(
                f"no pricing entry for {provider!r}/{model_id!r}; cost will be None",
                stacklevel=2,
            )
            return None
        in_cost = (tokens_in / 1_000_000.0) * rates.input_per_1m_tokens
        out_cost = (tokens_out / 1_000_000.0) * rates.output_per_1m_tokens
        reasoning_rate = (
            rates.reasoning_per_1m_tokens
            if rates.reasoning_per_1m_tokens is not None
            else rates.output_per_1m_tokens
        )
        reasoning_cost = (tokens_reasoning / 1_000_000.0) * reasoning_rate
        return in_cost + out_cost + reasoning_cost


def compute_cost(
    *,
    pricing: Pricing,
    provider: str,
    model_id: str,
    tokens_in: int,
    tokens_out: int,
    tokens_reasoning: int,
) -> float | None:
    return pricing.compute_cost(
        provider=provider,
        model_id=model_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tokens_reasoning=tokens_reasoning,
    )
