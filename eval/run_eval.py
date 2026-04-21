#!/usr/bin/env python
"""Batched LLM evaluation runner using vLLM offline inference.

Runs N episodes per environment with batched inference across all active
environments at each step. Uses vLLM's offline LLM class directly (no server).

Usage:
    uv run python eval/run_eval.py --model Qwen/Qwen3.5-4B --episodes 25
    uv run python eval/run_eval.py --config eval/configs/qwen35_4b.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import glyphbench  # noqa: F401 — register all envs
from glyphbench.core import all_glyphbench_env_ids
from glyphbench.core.base_env import BaseAsciiEnv
from glyphbench.harness.parser import parse_harness_output


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    turn: int
    observation: str
    action_name: str
    action_idx: int
    reward: float
    cumulative_reward: float
    terminated: bool
    truncated: bool
    parse_failed: bool
    llm_response: str


@dataclass
class EpisodeResult:
    env_id: str
    seed: int
    episode_idx: int
    total_reward: float = 0.0
    episode_length: int = 0
    parse_failures: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_s: float = 0.0
    trajectory: list[StepRecord] = field(default_factory=list)


@dataclass
class EnvResults:
    env_id: str
    episodes: list[EpisodeResult] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        returns = [e.total_reward for e in self.episodes]
        lengths = [e.episode_length for e in self.episodes]
        parse_fails = [e.parse_failures for e in self.episodes]
        return {
            "env_id": self.env_id,
            "n_episodes": len(self.episodes),
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "std_return": float(np.std(returns)) if returns else 0.0,
            "min_return": float(np.min(returns)) if returns else 0.0,
            "max_return": float(np.max(returns)) if returns else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "mean_parse_failures": float(np.mean(parse_fails)) if parse_fails else 0.0,
            "parse_failure_rate": float(np.sum(parse_fails)) / max(float(np.sum(lengths)), 1.0),
            "total_input_tokens": sum(e.total_input_tokens for e in self.episodes),
            "total_output_tokens": sum(e.total_output_tokens for e in self.episodes),
            "total_latency_s": sum(e.total_latency_s for e in self.episodes),
            "per_episode_returns": returns,
        }


# ---------------------------------------------------------------------------
# vLLM engine wrapper
# ---------------------------------------------------------------------------


class VLLMEngine:
    """Thin wrapper around vLLM's offline LLM class for batched chat inference."""

    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        gpu_memory_utilization: float = 0.9,
        enable_prefix_caching: bool = True,
        max_model_len: int = 16384,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = True,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vllm not installed. Install with: uv add 'glyphbench[eval]'"
            )

        engine_kwargs: dict[str, Any] = {
            "model": model_id,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_prefix_caching": enable_prefix_caching,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
        }
        if enforce_eager:
            engine_kwargs["enforce_eager"] = True

        print(f"Loading vLLM engine: {model_id}...")
        self._llm = LLM(**engine_kwargs)
        self._sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        print("Engine loaded.")

    def generate_batch(
        self, messages_list: list[list[dict[str, str]]]
    ) -> list[dict[str, Any]]:
        """Run batched chat inference. Returns list of {text, input_tokens, output_tokens}."""
        start = time.time()
        outputs = self._llm.chat(
            messages=messages_list,
            sampling_params=self._sampling_params,
        )
        latency = time.time() - start
        per_item_latency = latency / max(len(outputs), 1)

        results = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)
            results.append({
                "text": text,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "latency_s": per_item_latency,
            })
        return results


# ---------------------------------------------------------------------------
# Batched episode runner
# ---------------------------------------------------------------------------


@dataclass
class HarnessMode:
    """Controls how prompts are built for the LLM."""
    use_cot: bool = False        # Allow chain-of-thought in response
    history_len: int = 0         # 0 = markovian (single obs), N>0 = last N steps

    @property
    def name(self) -> str:
        hist = "history" if self.history_len > 0 else "markov"
        cot = "cot" if self.use_cot else "zeroshot"
        return f"{hist}_{cot}"


HARNESS_MODES = {
    "markov_zeroshot": HarnessMode(use_cot=False, history_len=0),
    "markov_cot": HarnessMode(use_cot=True, history_len=0),
    "history_zeroshot": HarnessMode(use_cot=False, history_len=5),
    "history_cot": HarnessMode(use_cot=True, history_len=5),
}


@dataclass
class StepHistory:
    """Tracks recent (obs, action, reward) tuples for non-markovian modes."""
    entries: list[tuple[str, str, float]] = field(default_factory=list)
    max_len: int = 5

    def add(self, obs: str, action_name: str, reward: float) -> None:
        self.entries.append((obs, action_name, reward))
        if len(self.entries) > self.max_len:
            self.entries.pop(0)

    def render(self) -> str:
        if not self.entries:
            return "(no history yet)"
        parts = []
        for i, (obs, act, rew) in enumerate(self.entries):
            # Truncate obs to grid section only for history (save tokens)
            grid_lines = []
            in_grid = False
            for line in obs.split("\n"):
                if line == "[Grid]":
                    in_grid = True
                    continue
                if line.startswith("[") and in_grid:
                    break
                if in_grid:
                    grid_lines.append(line)
            grid_text = "\n".join(grid_lines) if grid_lines else obs[:200]
            # Present the causal chain in natural temporal order: the obs the
            # model saw, then the action it chose from that obs, then the
            # reward received for that action. Putting action/reward FIRST
            # (as the earlier rendering did) invites the reader to interpret
            # the grid as a post-action resulting state, inverting the
            # causal chain. The stored tuple is (pre_action_obs, action, reward_for_action).
            parts.append(
                f"[Step {i+1}] Observed:\n{grid_text}\n"
                f"Chose action: {act} | Reward received: {rew:+.2f}"
            )
        return "\n\n".join(parts)


def build_messages(
    env: BaseAsciiEnv,
    obs: str,
    turn: int,
    mode: HarnessMode,
    history: StepHistory | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for one environment at one timestep.

    Supports 4 harness modes:
    - markov_zeroshot: single obs, just action
    - markov_cot: single obs, thinking + action
    - history_zeroshot: last N obs/action/rewards, just action
    - history_cot: last N obs/action/rewards, thinking + action
    """
    system = env.system_prompt()
    action_names = env.action_spec.names

    user_parts = []

    # History section (non-markovian modes)
    if mode.history_len > 0 and history and history.entries:
        user_parts.append("[Recent History]")
        user_parts.append(history.render())
        user_parts.append("")

    # Current observation
    user_parts.append(f"[Observation — Turn {turn}]")
    user_parts.append(obs)
    user_parts.append("")
    user_parts.append(f"Choose one action from: {action_names}")

    # Response format depends on CoT mode
    if mode.use_cot:
        user_parts.append(
            'Respond with JSON: {"thinking": "your reasoning", "action": "ACTION_NAME"}'
        )
    else:
        user_parts.append('Respond with JSON: {"action": "ACTION_NAME"}')

    user = "\n".join(user_parts)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def run_batched_episodes(
    engine: VLLMEngine,
    env_id: str,
    seeds: list[int],
    max_turns: int = 200,
    batch_size: int = 10,
    mode: HarnessMode | None = None,
) -> list[EpisodeResult]:
    """Run multiple episodes of one env with batched inference.

    All episodes step in lockstep — at each timestep, we batch all active
    environments' prompts into a single vLLM call.
    """
    if mode is None:
        mode = HARNESS_MODES["markov_zeroshot"]

    n = len(seeds)
    results: list[EpisodeResult] = []

    # Process in chunks of batch_size
    for chunk_start in range(0, n, batch_size):
        chunk_seeds = seeds[chunk_start : chunk_start + batch_size]
        chunk_n = len(chunk_seeds)

        # Create environments
        envs: list[BaseAsciiEnv] = []
        observations: list[str] = []
        for seed in chunk_seeds:
            raw_env = gym.make(env_id, max_turns=max_turns)
            env: BaseAsciiEnv = raw_env.unwrapped  # type: ignore[assignment]
            obs, _ = env.reset(seed=seed)
            envs.append(env)
            observations.append(obs)

        # Per-episode tracking
        active = [True] * chunk_n
        histories = [StepHistory(max_len=mode.history_len) for _ in range(chunk_n)]
        ep_results = [
            EpisodeResult(env_id=env_id, seed=chunk_seeds[i], episode_idx=chunk_start + i)
            for i in range(chunk_n)
        ]

        turn = 0
        while any(active) and turn < max_turns:
            turn += 1
            active_indices = [i for i in range(chunk_n) if active[i]]

            # Build messages for active envs (with harness mode)
            messages_batch = [
                build_messages(
                    envs[i], observations[i], turn, mode,
                    histories[i] if mode.history_len > 0 else None,
                )
                for i in active_indices
            ]

            # Batched inference
            responses = engine.generate_batch(messages_batch)

            # Process responses and step envs
            for idx_in_batch, env_idx in enumerate(active_indices):
                resp = responses[idx_in_batch]
                env = envs[env_idx]
                ep = ep_results[env_idx]

                # Track tokens
                ep.total_input_tokens += resp["input_tokens"]
                ep.total_output_tokens += resp["output_tokens"]
                ep.total_latency_s += resp["latency_s"]

                # Parse action
                parse_result = parse_harness_output(
                    resp["text"],
                    env.action_spec,
                    noop_action_name=env.noop_action_name,
                )
                if parse_result.fell_back_to_noop:
                    ep.parse_failures += 1
                action_idx = parse_result.action_index

                # Record pre-step observation
                pre_obs = observations[env_idx]

                # Step
                obs, reward, terminated, truncated, info = env.step(action_idx)
                observations[env_idx] = obs
                ep.total_reward += reward
                ep.episode_length += 1

                # Update history for non-markovian modes
                if mode.history_len > 0:
                    histories[env_idx].add(
                        pre_obs, parse_result.action_name, reward
                    )

                # Record trajectory step
                ep.trajectory.append(StepRecord(
                    turn=turn,
                    observation=pre_obs,
                    action_name=parse_result.action_name,
                    action_idx=action_idx,
                    reward=reward,
                    cumulative_reward=ep.total_reward,
                    terminated=terminated,
                    truncated=truncated,
                    parse_failed=parse_result.fell_back_to_noop,
                    llm_response=resp["text"],
                ))

                if terminated or truncated:
                    active[env_idx] = False
                    # Record final observation
                    ep.trajectory.append(StepRecord(
                        turn=turn + 1,
                        observation=obs,
                        action_name="DONE",
                        action_idx=-1,
                        reward=0.0,
                        cumulative_reward=ep.total_reward,
                        terminated=terminated,
                        truncated=truncated,
                        parse_failed=False,
                        llm_response="",
                    ))

        # Close envs
        for env in envs:
            env.close()

        results.extend(ep_results)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config file."""
    import yaml

    with path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="GlyphBench batched LLM evaluation")
    parser.add_argument("--config", type=Path, help="YAML config file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B", help="HF model ID")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per env")
    parser.add_argument(
        "--harness", type=str, default="markov_zeroshot",
        choices=list(HARNESS_MODES.keys()),
        help="Harness mode: markov_zeroshot, markov_cot, history_zeroshot, history_cot",
    )
    parser.add_argument("--history-len", type=int, default=5, help="History window for non-markovian modes")
    parser.add_argument("--max-turns", type=int, default=200, help="Max turns per episode")
    parser.add_argument("--batch-size", type=int, default=25, help="Batch size for inference")
    parser.add_argument("--output", type=Path, default=Path("results/"), help="Output directory")
    parser.add_argument("--envs", nargs="*", help="Specific env IDs (default: all)")
    parser.add_argument("--suites", nargs="*", help="Filter by suite (minigrid, minihack, etc.)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=16384, help="Max output tokens (16384 for generous thinking budget)")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--enforce-eager", action="store_true", default=True,
        help="Disable CUDA graphs (required for MIG GPU slices)",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    # Load config if provided, CLI args override
    if args.config and args.config.exists():
        cfg = load_config(args.config)
        for key, val in cfg.items():
            key_underscore = key.replace("-", "_")
            if hasattr(args, key_underscore) and getattr(args, key_underscore) is None:
                setattr(args, key_underscore, val)

    # Determine env list
    if args.envs:
        env_ids = args.envs
    else:
        env_ids = all_glyphbench_env_ids()
        if args.suites:
            env_ids = [
                e for e in env_ids
                if any(suite in e for suite in args.suites)
            ]

    # Filter out dummy envs
    env_ids = [e for e in env_ids if "dummy" not in e]

    # Set up harness mode
    mode = HarnessMode(
        use_cot=HARNESS_MODES[args.harness].use_cot,
        history_len=args.history_len if "history" in args.harness else 0,
    )

    print(f"Model: {args.model}")
    print(f"Harness: {mode.name} (history_len={mode.history_len})")
    print(f"Environments: {len(env_ids)}")
    print(f"Episodes per env: {args.episodes}")
    print(f"Max turns: {args.max_turns}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Initialize vLLM engine
    engine = VLLMEngine(
        model_id=args.model,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
    )

    # Generate seeds
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 2**31, size=args.episodes).tolist()

    # Output directory: results/{model}/{harness_mode}/
    model_slug = args.model.replace("/", "_")
    output_dir = args.output / model_slug / mode.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluations
    all_results: dict[str, dict[str, Any]] = {}
    total_start = time.time()

    for idx, env_id in enumerate(env_ids):
        env_start = time.time()
        print(f"[{idx + 1}/{len(env_ids)}] {env_id}...", end=" ", flush=True)

        try:
            episodes = run_batched_episodes(
                engine=engine,
                env_id=env_id,
                seeds=seeds,
                max_turns=args.max_turns,
                batch_size=args.batch_size,
                mode=mode,
            )
            env_results = EnvResults(env_id=env_id, episodes=episodes)
            summary = env_results.summary()
            all_results[env_id] = summary

            # Save per-env result
            env_slug = env_id.replace("/", "__")
            env_path = output_dir / "per_env" / f"{env_slug}.json"
            env_path.parent.mkdir(parents=True, exist_ok=True)
            with env_path.open("w") as f:
                json.dump(summary, f, indent=2)

            # Save trajectories
            traj_dir = output_dir / "trajectories" / env_slug
            traj_dir.mkdir(parents=True, exist_ok=True)
            for ep in episodes:
                traj_path = traj_dir / f"seed_{ep.seed}_ep_{ep.episode_idx}.jsonl"
                with traj_path.open("w") as f:
                    for step in ep.trajectory:
                        f.write(json.dumps({
                            "turn": step.turn,
                            "observation": step.observation,
                            "action_name": step.action_name,
                            "action_idx": step.action_idx,
                            "reward": step.reward,
                            "cumulative_reward": step.cumulative_reward,
                            "terminated": step.terminated,
                            "truncated": step.truncated,
                            "parse_failed": step.parse_failed,
                            "llm_response": step.llm_response,
                        }) + "\n")

            elapsed = time.time() - env_start
            print(
                f"return={summary['mean_return']:+.3f} "
                f"len={summary['mean_length']:.0f} "
                f"parse_fail={summary['parse_failure_rate']:.1%} "
                f"({elapsed:.1f}s)"
            )

        except Exception as e:
            print(f"ERROR: {e}")
            all_results[env_id] = {"env_id": env_id, "error": str(e)}

    total_elapsed = time.time() - total_start

    # Save aggregate results
    aggregate = {
        "model": args.model,
        "n_envs": len(env_ids),
        "episodes_per_env": args.episodes,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "total_time_s": total_elapsed,
        "per_env": all_results,
    }
    aggregate_path = output_dir / "results.json"
    with aggregate_path.open("w") as f:
        json.dump(aggregate, f, indent=2)

    # Print summary
    print(f"\nDone! {len(env_ids)} envs evaluated in {total_elapsed:.0f}s")
    print(f"Results saved to: {output_dir}")

    # Suite-level summary
    suite_stats: dict[str, list[float]] = {}
    for env_id, res in all_results.items():
        if "error" in res:
            continue
        suite = env_id.split("/")[1].split("-")[0] if "/" in env_id else "unknown"
        suite_stats.setdefault(suite, []).append(res["mean_return"])

    print("\nPer-suite mean returns:")
    for suite, returns in sorted(suite_stats.items()):
        print(f"  {suite:<12s}: {np.mean(returns):+.4f} ({len(returns)} envs)")


if __name__ == "__main__":
    main()
