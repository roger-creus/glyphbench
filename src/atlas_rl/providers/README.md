# atlas_rl.providers

Four LLM provider clients (vLLM, OpenAI, Anthropic, Gemini) implementing a shared async `LLMClient` protocol. Each client takes a pre-built SDK instance in its constructor for testability. `pricing.yaml` is the single source of truth for dollar-cost computation.

## Public types

- `LLMClient` — structural protocol.
- `LLMResponse` — re-exported from `harness.mock_client`.
- `build_client(config)` — factory that returns the right client for the config.
- `with_retries(coro_fn, ...)` — async retry/backoff utility.
- `Pricing` — loaded from `pricing.yaml`, computes dollar cost from tokens.

## Exception hierarchy

- `ProviderError` (base)
  - `ProviderRateLimit` — 429; retryable
  - `ProviderTransient` — 5xx / connection errors; retryable
  - `ProviderInvalidRequest` — 4xx non-retryable
  - `ProviderTimeout` — timeout; retryable

## Invariants

1. All four clients are async. Sync callers wrap with `asyncio.run`.
2. `dollar_cost` is always populated — either a float or `None` with a one-time warning (missing pricing).
3. vLLM clients always report `dollar_cost = 0.0`.
4. Retries are for provider-level errors only; parse errors are the harness's job.
