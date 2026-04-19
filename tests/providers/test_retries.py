import asyncio

import pytest

from atlas_rl.providers.base import ProviderRateLimit, ProviderTransient
from atlas_rl.providers.retries import with_retries


@pytest.mark.asyncio
async def test_first_call_succeeds_no_retry():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        return "ok"

    result = await with_retries(fn, max_retries=3, base_backoff_s=0.0)
    assert result == "ok"
    assert calls == 1


@pytest.mark.asyncio
async def test_retries_on_rate_limit_then_succeeds():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ProviderRateLimit("429")
        return "ok"

    result = await with_retries(fn, max_retries=5, base_backoff_s=0.0)
    assert result == "ok"
    assert calls == 3


@pytest.mark.asyncio
async def test_gives_up_after_max_retries_and_re_raises():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise ProviderTransient("5xx")

    with pytest.raises(ProviderTransient):
        await with_retries(fn, max_retries=2, base_backoff_s=0.0)
    # Initial call + 2 retries = 3 calls
    assert calls == 3


@pytest.mark.asyncio
async def test_non_retryable_error_short_circuits():
    class NotRetryable(Exception):
        pass

    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise NotRetryable("boom")

    with pytest.raises(NotRetryable):
        await with_retries(
            fn,
            max_retries=5,
            base_backoff_s=0.0,
            retryable_errors=(ProviderRateLimit, ProviderTransient),
        )
    assert calls == 1


@pytest.mark.asyncio
async def test_backoff_is_called(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ProviderRateLimit("429")
        return "ok"

    await with_retries(fn, max_retries=5, base_backoff_s=1.0, max_backoff_s=60.0)
    # Expect two sleeps (between attempts 1->2 and 2->3)
    assert len(sleeps) == 2
    # Exponential: first sleep ~1s, second ~2s (we allow jitter tolerance later)
    assert sleeps[0] >= 1.0
    assert sleeps[1] >= 2.0
