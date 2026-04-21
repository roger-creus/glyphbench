"""Public types for the providers subpackage: LLMClient protocol, LLMResponse, errors."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from glyphbench.harness.mock_client import LLMResponse

__all__ = [
    "LLMClient",
    "LLMResponse",
    "ProviderError",
    "ProviderRateLimit",
    "ProviderTransient",
    "ProviderInvalidRequest",
    "ProviderTimeout",
]


class ProviderError(Exception):
    """Base class for every provider-level error."""


class ProviderRateLimit(ProviderError):
    """HTTP 429 or provider-specific rate-limit signal. Retryable."""


class ProviderTransient(ProviderError):
    """5xx, connection reset, etc. Retryable."""


class ProviderInvalidRequest(ProviderError):
    """4xx that is not rate-limit. Not retryable - usually a bug."""


class ProviderTimeout(ProviderError):
    """Request timed out. Retryable."""


@runtime_checkable
class LLMClient(Protocol):
    model_id: str
    provider: str

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        response_format: dict[str, Any] | None,
        seed: int | None,
    ) -> LLMResponse: ...
