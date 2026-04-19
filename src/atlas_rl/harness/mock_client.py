"""MockLLMClient: a deterministic in-memory LLM client used as a test fixture.

Used by harness integration tests (this plan) and by the runner and pilot env
plans (0.C, 0.D) as the default "LLM" in end-to-end tests. Real provider
clients live in `atlas_rl.providers` and implement the same protocol.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Matches the protocol used by the runner. Fully defined here; providers
    will import this type from `providers.base` later — for now this is the
    single source of truth and providers/base will re-export it."""

    text: str
    tokens_in: int
    tokens_out: int
    tokens_reasoning: int
    dollar_cost: float | None
    latency_s: float
    provider: str
    provider_request_id: str | None
    raw_response: dict[str, Any]


@dataclass
class ScriptedResponse:
    text: str
    tokens_in: int = 100
    tokens_out: int = 20
    tokens_reasoning: int = 0
    latency_s: float = 0.0


class MockLLMClient:
    """Deterministic LLM client. Returns scripted responses in order, or always
    the same response if constructed via `MockLLMClient.always(text)`.
    """

    model_id = "mock-model"
    provider = "mock"

    def __init__(self, scripted: list[ScriptedResponse], *, always: bool = False) -> None:
        if always:
            assert len(scripted) == 1
            self._always = True
            self._always_response = scripted[0]
            self._queue: deque[ScriptedResponse] = deque()
        else:
            self._always = False
            self._queue = deque(scripted)
            self._always_response = None  # type: ignore[assignment]
        self._calls = 0

    @classmethod
    def always(cls, text: str, *, tokens_in: int = 100, tokens_out: int = 20) -> MockLLMClient:
        return cls(
            scripted=[ScriptedResponse(text=text, tokens_in=tokens_in, tokens_out=tokens_out)],
            always=True,
        )

    @property
    def call_count(self) -> int:
        return self._calls

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
        response_format: dict[str, Any] | None,
        seed: int | None,
    ) -> LLMResponse:
        self._calls += 1
        if self._always:
            r = self._always_response
        else:
            if not self._queue:
                raise RuntimeError("MockLLMClient script exhausted")
            r = self._queue.popleft()
        return LLMResponse(
            text=r.text,
            tokens_in=r.tokens_in,
            tokens_out=r.tokens_out,
            tokens_reasoning=r.tokens_reasoning,
            dollar_cost=0.0,
            latency_s=r.latency_s,
            provider="mock",
            provider_request_id=f"mock-{self._calls}",
            raw_response={"system_prompt": system_prompt, "user_prompt": user_prompt},
        )
