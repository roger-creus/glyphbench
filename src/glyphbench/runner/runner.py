"""The async benchmark runner: parallel episodes, storage, dashboard, budget."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from typing import Any

import gymnasium as gym

from glyphbench.core.base_env import BaseAsciiEnv
from glyphbench.harness.agent import HarnessAgent
from glyphbench.providers.base import LLMClient
from glyphbench.runner.random_agent import RandomAgent
from glyphbench.providers.factory import ClientBuildConfig, build_client
from glyphbench.providers.pricing import Pricing
from glyphbench.runner.budget import BudgetExceeded, CostTracker
from glyphbench.runner.config import RunConfig
from glyphbench.runner.dashboard import Dashboard, NullDashboard
from glyphbench.runner.storage import EpisodeRecord, RunStorage


def _derive_episode_seed(env_id: str, seed: int, episode_idx: int) -> int:
    key = f"{env_id}|{seed}|{episode_idx}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big")


def _default_client_factory(
    config: RunConfig, pricing: Pricing
) -> Callable[[], LLMClient]:
    def factory() -> LLMClient:
        if config.provider == "mock":
            raise ValueError(
                "mock provider requires an explicit client_factory; "
                "the runner does not know how to build a MockLLMClient on its own"
            )
        cfg = ClientBuildConfig(
            provider=config.provider,
            model_id=config.model_id,
            base_url=config.base_url,
        )
        return build_client(cfg, pricing=pricing)

    return factory


async def run_benchmark(
    config: RunConfig,
    *,
    client_factory: Callable[[], LLMClient] | None = None,
) -> None:
    """Run a benchmark according to `config`.

    `client_factory` is a zero-arg callable returning a fresh LLMClient. Useful
    for tests (injecting MockLLMClient) and for providers that should not share
    state across concurrent episodes. If None, a factory is built from the
    config + pricing.
    """
    import glyphbench  # noqa: F401 — trigger env registration

    pricing = Pricing.from_yaml(config.pricing_yaml)
    if client_factory is None:
        client_factory = _default_client_factory(config, pricing)

    storage = RunStorage(
        base_dir=config.output_dir,
        run_id=config.run_id,
        trajectory_logging=config.harness.trajectory_logging,
    )
    dashboard: Dashboard | NullDashboard
    total_episodes = len(config.envs) * len(config.seeds) * config.episodes_per_env
    if config.dashboard:
        dashboard = Dashboard(
            run_id=config.run_id,
            model_id=config.model_id,
            provider=config.provider,
            total_episodes=total_episodes,
        )
    else:
        dashboard = NullDashboard()

    cost_tracker = CostTracker(budget_usd=config.budget_usd)
    semaphore = asyncio.Semaphore(config.concurrency)
    lock = asyncio.Lock()

    # Running aggregates per env for dashboard.
    env_stats: dict[str, dict[str, Any]] = {
        env_id: {
            "returns": [],
            "lengths": [],
            "done": 0,
            "total": len(config.seeds) * config.episodes_per_env,
        }
        for env_id in config.envs
    }

    budget_event = asyncio.Event()

    dashboard.start()
    try:
        tasks: list[asyncio.Task[None]] = []
        for env_id in config.envs:
            for seed in config.seeds:
                for ep_idx in range(config.episodes_per_env):
                    tasks.append(
                        asyncio.create_task(
                            _run_episode(
                                env_id=env_id,
                                seed=seed,
                                episode_idx=ep_idx,
                                config=config,
                                client_factory=client_factory,
                                storage=storage,
                                dashboard=dashboard,
                                cost_tracker=cost_tracker,
                                semaphore=semaphore,
                                env_stats=env_stats,
                                lock=lock,
                                budget_event=budget_event,
                            )
                        )
                    )
        await asyncio.gather(*tasks)
    finally:
        storage.finalize()
        dashboard.stop()


async def _run_episode(
    *,
    env_id: str,
    seed: int,
    episode_idx: int,
    config: RunConfig,
    client_factory: Callable[[], LLMClient],
    storage: RunStorage,
    dashboard: Dashboard | NullDashboard,
    cost_tracker: CostTracker,
    semaphore: asyncio.Semaphore,
    env_stats: dict[str, dict[str, Any]],
    lock: asyncio.Lock,
    budget_event: asyncio.Event,
) -> None:
    async with semaphore:
        if budget_event.is_set():
            return
        episode_seed = _derive_episode_seed(env_id, seed, episode_idx)
        raw_env = gym.make(env_id, max_turns=config.max_turns_per_episode)
        env: BaseAsciiEnv = raw_env.unwrapped  # type: ignore[assignment]

        if config.provider == "random":
            agent: HarnessAgent | RandomAgent = RandomAgent(
                env=env, seed=episode_seed,
            )
        else:
            client = client_factory()
            agent = HarnessAgent(
                env=env,
                client=client,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                model_seed=config.model_seed,
            )

        episode_return, length, turn_metrics = await agent.run_episode(
            seed=episode_seed
        )

        terminated_reason = "success" if episode_return > 0 else "timeout"
        if turn_metrics and turn_metrics[-1].terminated:
            terminated_reason = "terminated"

        total_cost = sum(m.dollar_cost_turn for m in turn_metrics)
        try:
            cost_tracker.add(total_cost)
        except BudgetExceeded:
            budget_event.set()
            dashboard.log_event(f"[{env_id}] budget exceeded after episode complete")

        record = EpisodeRecord(
            env_id=env_id,
            seed=seed,
            episode_idx=episode_idx,
            episode_return=episode_return,
            episode_length=length,
            terminated_reason=terminated_reason,
            turn_metrics=turn_metrics,
            extras={},
        )
        async with lock:
            storage.record_episode(record)
            stats = env_stats[env_id]
            stats["returns"].append(episode_return)
            stats["lengths"].append(length)
            stats["done"] += 1
            mean_return = sum(stats["returns"]) / len(stats["returns"])
            mean_len = sum(stats["lengths"]) / len(stats["lengths"])
            dashboard.update_env(
                env_id,
                episodes_done=stats["done"],
                episodes_total=stats["total"],
                mean_return=mean_return,
                mean_len=mean_len,
            )
            dashboard.update_totals(
                cost_used=cost_tracker.total,
                budget=cost_tracker.budget,
                parse_failures=sum(
                    1 for m in turn_metrics if m.action_parse_error
                ),
                provider_errors=0,
                fallback_noops=sum(
                    1 for m in turn_metrics if m.action_fell_back_to_noop
                ),
            )
