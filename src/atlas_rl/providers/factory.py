"""Build an LLMClient from a minimal config.

This is the single place where SDK instantiation happens. Tests can bypass this
and instantiate clients directly with fake SDK instances.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from atlas_rl.providers.anthropic_client import AnthropicClient
from atlas_rl.providers.base import LLMClient
from atlas_rl.providers.gemini_client import GeminiClient
from atlas_rl.providers.openai_client import OpenAIClient
from atlas_rl.providers.pricing import Pricing
from atlas_rl.providers.vllm_client import VllmClient


@dataclass
class ClientBuildConfig:
    provider: str
    model_id: str
    base_url: str | None = None
    api_key_env: str | None = None  # override env var name (otherwise provider-standard)
    timeout_s: float = 120.0


def build_client(config: ClientBuildConfig, *, pricing: Pricing) -> LLMClient:
    provider = config.provider.lower()
    if provider == "vllm":
        if not config.base_url:
            raise ValueError("vllm provider requires base_url")
        http = httpx.AsyncClient(base_url=config.base_url, timeout=config.timeout_s)
        return VllmClient(model_id=config.model_id, http_client=http, pricing=pricing)

    if provider == "openai":
        import openai

        api_key = os.environ.get(config.api_key_env or "OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in env")
        openai_sdk = openai.AsyncOpenAI(api_key=api_key, timeout=config.timeout_s)
        return OpenAIClient(sdk_client=openai_sdk, model_id=config.model_id, pricing=pricing)

    if provider == "anthropic":
        import anthropic

        api_key = os.environ.get(config.api_key_env or "ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in env")
        anthropic_sdk = anthropic.AsyncAnthropic(api_key=api_key, timeout=config.timeout_s)
        return AnthropicClient(
            sdk_client=anthropic_sdk, model_id=config.model_id, pricing=pricing
        )

    if provider in ("google", "gemini"):
        from google import genai

        api_key = os.environ.get(config.api_key_env or "GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in env")
        gemini_sdk = genai.Client(api_key=api_key)
        return GeminiClient(sdk_client=gemini_sdk, model_id=config.model_id, pricing=pricing)

    raise ValueError(f"unknown provider: {config.provider!r}")
