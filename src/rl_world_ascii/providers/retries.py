"""Async retry/backoff utility for provider calls."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from rl_world_ascii.providers.base import (
    ProviderRateLimit,
    ProviderTimeout,
    ProviderTransient,
)

T = TypeVar("T")

DEFAULT_RETRYABLE: tuple[type[Exception], ...] = (
    ProviderRateLimit,
    ProviderTransient,
    ProviderTimeout,
)


async def with_retries(
    coro_fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 5,
    base_backoff_s: float = 1.0,
    max_backoff_s: float = 60.0,
    retryable_errors: tuple[type[Exception], ...] = DEFAULT_RETRYABLE,
) -> T:
    """Call `coro_fn` with exponential backoff on retryable errors.

    Up to `max_retries` retries after the initial call (so max `max_retries+1`
    total attempts). Backoff is `min(base * 2**attempt + jitter, max_backoff_s)`.
    """
    attempt = 0
    while True:
        try:
            return await coro_fn()
        except retryable_errors:
            if attempt >= max_retries:
                raise
            backoff = min(base_backoff_s * (2**attempt), max_backoff_s)
            jitter = random.uniform(0, backoff * 0.1)
            await asyncio.sleep(backoff + jitter)
            attempt += 1
