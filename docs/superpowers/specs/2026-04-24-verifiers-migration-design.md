# GlyphBench → Verifiers Migration Design

**Date:** 2026-04-24
**Branch:** `verifiers`
**Status:** Approved design; plan + implementation to follow in-session.

## Goal

Rebuild GlyphBench's environment infrastructure on top of [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) and wire it end-to-end with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) for evaluation and training. All 292 games keep their existing mechanics, rewards, and deterministic seeding; only the infrastructure layer (base class, registry, eval runner, providers, harness, Docker) is replaced.

**No gymnasium anywhere in the codebase after this migration.** No `gym.Env` inheritance, no `gym.register`, no `import gymnasium`.

**No `ascii` in code names either.** The project is `glyphbench`. Observations are Unicode glyphs, not ASCII. Naming is renamed throughout (`BaseAsciiEnv` → `BaseGlyphEnv`, `ascii_primitives.py` → `glyph_primitives.py`, doc strings updated).

## Background (what exists today)

- 292 games across 6 suites (`minigrid`, `minihack`, `atari`, `craftax`, `procgen`, `classics`), each a `gymnasium.Env` subclass of `BaseAsciiEnv`.
- Custom `harness/` (prompt builder, JSON parser), `providers/` (vLLM/OpenAI/Anthropic/Gemini clients), `runner/` (batched inference orchestrator), `eval/run_eval.py` (vLLM offline batched runner with 4 harness modes).
- Previous Dockerfile: `FROM vllm/vllm-openai:latest` + `pip install gymnasium pydantic ...`. To be fully replaced.
- 2108+ tests; test suite is green on `master`. Leaderboard was recently wiped — no backward-compat constraint with old result JSONs.

## Non-goals

- Per-env hub publishing via `prime env push`. Keep everything in one umbrella package.
- Multi-modal / VLM support.
- Re-running the old leaderboard matrix.
- Cluster-manager changes (`cluster_manager/` is internal, gitignored, unaffected).
- Preserving the old 4-harness-mode matrix (markov/history × zeroshot/cot). We ship **one** harness: frame-stacked history with chain-of-thought.

## Architecture

### Package layout after migration

```
src/glyphbench/
    __init__.py                       # re-exports load_environment, make_env, REGISTRY
    cli.py                            # current CLI; thin updates
    core/
        __init__.py
        base_env.py                   # BaseGlyphEnv (plain class, no gym.Env)
        action.py                     # ActionSpec — unchanged
        observation.py                # GridObservation — unchanged
        glyph_primitives.py           # renamed from ascii_primitives.py
        registry.py                   # REGISTRY: dict[str, type[BaseGlyphEnv]], register_env(id, cls), make_env(id, **kw)
        metrics.py                    # existing
    envs/
        {minigrid,minihack,atari,craftax,procgen,classics,dummy}/
                                      # all 292 game classes; imports swap BaseAsciiEnv→BaseGlyphEnv;
                                      # __init__.py swaps gym.register → register_env (class-object arg, not entry_point string)
    verifiers_integration/
        __init__.py                   # load_environment(...)
        env.py                        # GlyphbenchMultiTurnEnv(vf.MultiTurnEnv)
        parser.py                     # GlyphbenchXMLParser(vf.XMLParser)
        prompting.py                  # build_system_prompt(), render_user_turn(), frame-stack formatter
        rubric.py                     # EpisodicReturnRubric + monitor metrics
configs/
    eval/
        glyphbench-debug.toml         # prime-rl eval config, 2 envs × 2 episodes, Qwen3-0.6B
        glyphbench-full.toml          # all 292 envs × 10 episodes, Qwen3-0.6B
    rl/
        glyphbench-smoke/
            train.toml
            orch.toml
            infer.toml
eval/
    README.md                         # rewritten for vf-eval / prime eval run flow
    random_baseline.py                # ported to non-gym API
    random_baseline.json              # regenerated under new harness
scripts/
    demo_all_envs.py, play_random.py, play_interactive.py, play_curses.py,
    replay_trajectory.py, record_random_gifs.py, run_benchmark.py
                                      # all ported to non-gym API via make_env()
Dockerfile                            # nvidia/cuda:12.4.1-devel-ubuntu22.04 + uv + vllm + verifiers + prime-rl + glyphbench (editable)
scripts/build_sif.sh                  # docker build → apptainer build glyphbench.sif
tests/                                # every gym.make() → make_env(); new tests for verifiers wrapper + parser + prompting
pyproject.toml                        # gymnasium DROPPED; verifiers + datasets added; optional-extras [eval,analysis,dev] kept
```

### Deleted directories

- `src/glyphbench/harness/` — entire directory (agent.py, parser.py, prompt_builder.py, schema.py, state.py, templating.py, mock_client.py, system_prompts/, README.md).
- `src/glyphbench/providers/` — entire directory (anthropic_client.py, base.py, factory.py, gemini_client.py, openai_client.py, pricing.py, retries.py, vllm_client.py, README.md).
- `src/glyphbench/runner/` — entire directory (budget.py, config.py, dashboard.py, random_agent.py, runner.py, storage.py, README.md).
- `eval/run_eval.py`, `eval/scoring.py` (verifiers produces rewards natively; scoring for paper aggregation can be re-added later if needed — out of scope for this migration).
- `scripts/serve_vllm.sh` — vLLM serve command documented in the new `eval/README.md`.
- Old `Dockerfile` — fully replaced.

### Core (non-gym) base class

```python
# src/glyphbench/core/base_env.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation


class BaseGlyphEnv(ABC):
    """Base class for every game in glyphbench.

    Plain Python class — no framework inheritance. Subclasses implement the
    five abstract methods; the public reset/step surface is fixed here.
    """

    action_spec: ActionSpec              # class attribute set by subclass
    noop_action_name: str = "NOOP"

    def __init__(self, max_turns: int = 500) -> None:
        self.max_turns = max_turns
        self._turn: int = 0
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int) -> tuple[str, dict[str, Any]]:
        """Seed-required reset. Returns (rendered_obs, info)."""
        if not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be int, got {type(seed).__name__}")
        self._rng = np.random.default_rng(int(seed))
        self._turn = 0
        obs = self._reset(int(seed))
        info: dict[str, Any] = {"turn": 0, "env_id": self.env_id(), "seed": int(seed)}
        return obs.render(), info

    def step(self, action: int) -> tuple[str, float, bool, bool, dict[str, Any]]:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError(f"action must be int, got {type(action).__name__}")
        if not 0 <= int(action) < self.action_spec.n:
            raise ValueError(f"action {action} out of range [0, {self.action_spec.n})")
        self._turn += 1
        obs, reward, terminated, truncated, info = self._step(int(action))
        if self._turn >= self.max_turns and not (terminated or truncated):
            truncated = True
            info["truncation_reason"] = "max_turns"
        info["turn"] = self._turn
        info["env_id"] = self.env_id()
        return obs.render(), float(reward), terminated, truncated, info

    @abstractmethod
    def _reset(self, seed: int) -> GridObservation: ...
    @abstractmethod
    def _step(self, action: int) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]: ...
    @abstractmethod
    def _render_current_observation(self) -> GridObservation: ...
    @abstractmethod
    def system_prompt(self) -> str: ...    # describes the GAME only (task, rules, actions, reward)
    @abstractmethod
    def env_id(self) -> str: ...

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            raise RuntimeError("call reset() before accessing rng")
        return self._rng

    def get_observation(self) -> GridObservation:
        return self._render_current_observation()

    def close(self) -> None:
        """Optional cleanup hook. Default: no-op."""
        return None
```

**Differences vs. today's `BaseAsciiEnv`:**

- No `gym.Env[str, int]` inheritance.
- No `action_space` / `observation_space` / `metadata`.
- `reset(seed)` — positional int (not keyword-only).
- `close()` is a no-op default (was inherited from gym).
- Everything else — `ActionSpec`, `GridObservation`, subclass hooks, RNG handling — is bit-identical.

### Registry

```python
# src/glyphbench/core/registry.py
from __future__ import annotations
from typing import Any
from glyphbench.core.base_env import BaseGlyphEnv

REGISTRY: dict[str, type[BaseGlyphEnv]] = {}


def register_env(env_id: str, cls: type[BaseGlyphEnv]) -> None:
    if env_id in REGISTRY and REGISTRY[env_id] is not cls:
        raise ValueError(f"env_id {env_id!r} already registered to {REGISTRY[env_id].__name__}")
    REGISTRY[env_id] = cls


def make_env(env_id: str, **kwargs: Any) -> BaseGlyphEnv:
    if env_id not in REGISTRY:
        raise KeyError(f"unknown env_id {env_id!r}; known ids: {sorted(REGISTRY)[:5]}…")
    return REGISTRY[env_id](**kwargs)


def all_glyphbench_env_ids() -> list[str]:
    return sorted(REGISTRY)
```

Per-suite `__init__.py` files swap `register_env("id", "module.path:Class")` → `register_env("id", Class)` (class object, not entry-point string). Imports of the game modules happen eagerly in the suite's `__init__.py`; importing `glyphbench.envs` populates `REGISTRY` for all 292 games, same as today.

### Verifiers integration

```python
# src/glyphbench/verifiers_integration/__init__.py
from glyphbench.verifiers_integration.env import GlyphbenchMultiTurnEnv, load_environment

__all__ = ["GlyphbenchMultiTurnEnv", "load_environment"]
```

```python
# src/glyphbench/verifiers_integration/env.py

def load_environment(
    env_id: str | list[str] | None = None,
    num_episodes: int = 10,
    n_frames: int = 4,
    max_turns: int | None = None,
    max_output_tokens: int = 512,
    seed: int = 42,
    **kwargs: Any,
) -> vf.Environment:
    """Entry point consumed by `vf-eval` and `prime eval run`.

    Args:
        env_id: single env id, list of ids, or None for all registered envs
                (dummy envs excluded).
        num_episodes: episodes (= rollouts) per env.
        n_frames: history window shown in each user turn (N most recent steps).
        max_turns: per-episode turn cap; None = use the env's own max_turns.
        max_output_tokens: per-turn LLM budget; communicated to the model in the
                          response-format block of the system prompt.
        seed: base seed; each episode gets seed = base + episode_idx.
    """
    env_ids = _resolve_env_ids(env_id)
    dataset = _build_dataset(env_ids, num_episodes, seed)
    parser = GlyphbenchXMLParser()
    rubric = EpisodicReturnRubric(parser=parser)
    system_prompt_builder = lambda game: build_system_prompt(game, max_output_tokens)
    return GlyphbenchMultiTurnEnv(
        dataset=dataset,
        rubric=rubric,
        parser=parser,
        n_frames=n_frames,
        max_turns=max_turns,
        system_prompt_builder=system_prompt_builder,
    )


class GlyphbenchMultiTurnEnv(vf.MultiTurnEnv):
    async def setup_state(self, state):
        info = json.loads(state["info"]) if isinstance(state["info"], str) else state["info"]
        env_id = info["env_id"]
        seed = info["seed"]
        game = make_env(env_id, max_turns=self._max_turns_override or 500)
        obs_text, _ = game.reset(seed)
        state["game"] = game
        state["frames"] = deque(maxlen=self.n_frames)   # (obs_text_pre_action, action_name, reward)
        state["current_obs"] = obs_text
        state["done"] = False
        state["parse_failures"] = 0
        state["episode_return"] = 0.0
        state["terminated"] = False
        state["truncated"] = False
        return await super().setup_state(state)

    async def env_response(self, messages, state, **kwargs):
        game = state["game"]
        raw = messages[-1].get("content", "") if messages else ""
        action_idx, action_name, parse_failed = self.parser.parse_action(raw, game)
        if parse_failed:
            state["parse_failures"] += 1

        pre_obs = state["current_obs"]
        obs_text, reward, term, trunc, info = game.step(action_idx)
        state["frames"].append((pre_obs, action_name, reward))
        state["current_obs"] = obs_text
        state["episode_return"] += reward
        state["terminated"] = bool(term)
        state["truncated"] = bool(trunc)
        state["done"] = bool(term or trunc)
        state["trajectory"][-1]["reward"] = reward

        next_user = render_user_turn(game, state["frames"], obs_text, turn=game._turn)
        return [{"role": "user", "content": next_user}]

    @vf.stop
    async def is_done(self, state):
        return state.get("done", False)

    @vf.cleanup
    async def close_game(self, state):
        game = state.pop("game", None)
        if game is not None:
            try: game.close()
            except Exception: pass
```

### Frame-stack prompting (the "flawless LLM prompt" spec)

All prompt construction lives in `verifiers_integration/prompting.py`. The principles:

1. **One legend per user message, globally deduped.** The legend changes rarely within an episode; rendering it 4× (history) + 1× (current) wastes tokens. We render the legend **once**, at the top of the user message, as the union of all symbols present in (frames ∪ current_obs). History frames and the current observation are stripped of their `[Legend]` section before being embedded.
2. **Stable section structure.** Every user message has the exact same section order: `[Legend] → [History] → [Current Observation] → [Actions] → [Format reminder]`. This maximizes KV-cache prefix overlap across turns and is friendly to vLLM's prefix caching.
3. **History rendered in temporal order.** Each history entry is rendered as: pre-action observation (grid only, no legend, no HUD duplicate), then the action taken, then the reward received. This preserves the causal chain the existing code already figured out (see the comment block in `eval/run_eval.py` lines 220-230 of the old code — we keep that invariant).
4. **Grid-only for history frames.** To save tokens, history entries strip `[Legend]` (emitted once at top) and collapse `[HUD]` into a one-line delta (`step=X, score=Y, carrying=Z`) rather than the full multi-line HUD block.
5. **Current observation kept complete.** The last `[Current Observation]` section shows the full grid + HUD + optional `[Message]`. Only the legend is removed (since it's at the top).
6. **Actions enumerated by name.** `[Actions] Choose one: [LEFT, RIGHT, PICKUP, ...]`.
7. **Format block reiterated.** Short reminder at the bottom of every user message: `Respond with <think>... (≤ 480 tokens)</think>\n<action>NAME</action>. Total response budget: 512 tokens.` — this constant budget hint prevents runaway thinking.
8. **System prompt owns the game rules.** Game-specific text (task, rules, reward structure) comes from `game.system_prompt()`. The harness appends a generic response-format block describing `<think>`/`<action>` XML tags, the 512-token budget, and the noop-fallback behavior.

System prompt structure (built once per rollout):

```
<game.system_prompt()>

---
RESPONSE FORMAT
Respond with exactly two XML tags:
  <think>your reasoning here — keep it concise; this is where you plan your move</think>
  <action>ACTION_NAME</action>

Your total response budget is 512 tokens (thinking + action combined). Any text outside these two tags is ignored. If your <action> tag is missing or contains an unknown action name, the NOOP action is applied instead.

ACTIONS AVAILABLE: [TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, …]
(with per-action descriptions from action_spec)
```

User-turn template (emitted every turn):

```
[Legend]
  █ wall
  · floor
  ★ goal
  → you, facing right
  ...
  (only symbols present in this turn's frames + current obs)

[History — last 4 turns]
  (turn T-4) Observed:
    <grid-only rendering, no legend>
    step=X, score=Y, HP=Z (HUD delta line)
  → chose ACTION_NAME → reward +0.00

  (turn T-3) …
  (turn T-2) …
  (turn T-1) …

[Current Observation — turn T]
  <full grid>
  <full HUD>
  [Message] <per-turn narrative if any>

[Actions]
  Choose one: [TURN_LEFT, TURN_RIGHT, ...]

Reply with <think>…</think><action>NAME</action>. 512-token total budget.
```

The first turn has an empty history block which collapses entirely (no `[History]` section emitted), so turn 0 user messages are `[Legend] → [Current Observation] → [Actions] → Format`.

### Parser

`GlyphbenchXMLParser(vf.XMLParser)` with `fields=["think", "action"]`, `answer_field="action"`. Exposes `parse_action(raw_text, game) -> (action_idx, action_name, parse_failed)`:

- Use verifiers' XML extraction first; fall back to a brace-balanced scan if the model emits JSON (tolerance for models that don't follow XML instructions).
- Normalize: strip, uppercase-compare against `action_spec.names`.
- Unknown / missing → `(noop_idx, noop_action_name, True)`.

Tests cover: well-formed XML, missing `<action>`, unknown action name, malformed XML, model emitting JSON fallback, multiple `<action>` tags (take the last).

### Rubric

```python
class EpisodicReturnRubric(vf.Rubric):
    def __init__(self, parser=None, **kw):
        super().__init__(parser=parser, **kw)
        self.add_reward_func(self.episodic_return, weight=1.0)
        self.add_metric(self.parse_failure_rate)
        self.add_metric(self.episode_length)
        self.add_metric(self.terminated_flag)
        self.add_metric(self.truncated_flag)
        fmt = parser.get_format_reward_func() if parser else None
        if fmt:
            fmt.__name__ = "xml_format_reward"
            self.add_metric(fmt)

    async def episodic_return(self, state) -> float:
        return float(state.get("episode_return", 0.0))

    async def parse_failure_rate(self, state) -> float:
        traj_len = max(len(state.get("trajectory", [])), 1)
        return state.get("parse_failures", 0) / traj_len

    async def episode_length(self, state) -> float:
        return float(len(state.get("trajectory", [])))

    async def terminated_flag(self, state) -> float:
        return 1.0 if state.get("terminated") else 0.0

    async def truncated_flag(self, state) -> float:
        return 1.0 if state.get("truncated") else 0.0
```

### Dataset shape

One row per (env_id × episode_idx). Row schema:

```
{
    "info": json.dumps({"env_id": env_id, "seed": base_seed + episode_idx}),
    "prompt": [
        {"role": "system", "content": <built-from-game-instance-lazily-in-setup_state>},
        {"role": "user",   "content": <initial observation render at turn 0>},
    ],
    "answer": "",
}
```

Because the system/user content depends on the per-rollout game instance, we build the `prompt` lazily in `setup_state` — the dataset row carries only `info`, and `setup_state` populates `state["prompt"]` after instantiating the game and calling `reset(seed)`. The `prompt` column in the dataset is a static placeholder ready to be overridden.

(Verifiers supports dynamic prompts via `setup_state` writing to `state["prompt"]`.)

### Sampling defaults

- `temperature=0.7`, `top_p=0.9`, `max_tokens=512`.
- Exposed as `load_environment(max_output_tokens=512, temperature=0.7, top_p=0.9)`.
- `prime eval run` / `vf-eval -t 512` also overrides.

## Data flow — one rollout

1. `vf-eval glyphbench -m Qwen/Qwen3-0.6B -x '{"env_id":"glyphbench/minigrid-empty-5x5-v0","num_episodes":4}' -t 512`
2. `load_environment()` resolves the env id, builds a 4-row dataset, returns a `GlyphbenchMultiTurnEnv`.
3. For each row, verifiers calls `setup_state(state)`:
    - parses `info.env_id`, `info.seed`
    - instantiates the game via `make_env(env_id)`
    - calls `game.reset(seed)` → initial observation text
    - populates `state["prompt"]` with system prompt (from game) + first user turn (initial obs, empty history)
    - initializes `state["frames"]` (deque, maxlen=4), `state["current_obs"]`, `state["episode_return"]`, `state["parse_failures"]`, `state["done"]=False`
4. Verifiers' rollout loop:
    - `get_prompt_messages` assembles conversation
    - model generates response (≤ 512 tokens)
    - `env_response(messages, state)`: parse, step, append to frames, render next user turn, update state
    - `is_done` returns `state["done"]`; loop exits on terminal / truncated
5. Rubric scores: `episodic_return`, plus metrics (`parse_failure_rate`, `episode_length`, `terminated_flag`, `truncated_flag`, `xml_format_reward`).
6. Verifiers writes results + trajectories under `~/.prime/evals/...` (default); viewable via `prime eval tui`.

## Tests

### Porting the existing 2108+ tests

The mechanical swap is small:

- `import gymnasium as gym` / `import gym` → removed.
- `gym.make("glyphbench/xxx")` → `make_env("glyphbench/xxx")`.
- Calls that accessed `env.unwrapped` → just `env` (no wrapper).
- Assertions on `env.action_space.n` → `env.action_spec.n`.
- Assertions on `env.observation_space` — removed (gymnasium-only).
- `env.reset(seed=42)` → `env.reset(42)` (positional).

All game-logic assertions (reward values, termination conditions, step counts, grid contents, system-prompt contents) are unchanged. The `hypothesis`-based property tests on env invariants port unchanged.

### New tests

- `tests/core/test_registry.py` — registration, lookup, duplicate detection, `make_env` kwargs.
- `tests/verifiers_integration/test_parser.py` — 12+ parse-failure classes (well-formed, missing tag, unknown action, JSON fallback, multiple action tags, case insensitivity, whitespace, …).
- `tests/verifiers_integration/test_prompting.py` — legend dedup, history window = n_frames, grid-only history rendering, turn-0 empty-history collapse, section ordering invariance.
- `tests/verifiers_integration/test_env.py` — `load_environment` dataset shape; `setup_state` populates `state["game"]` + initial obs; `env_response` steps the game, updates frames, updates episode_return; `is_done` transitions on termination; `cleanup` closes game.
- `tests/verifiers_integration/test_rubric.py` — `episodic_return` sums per-step rewards; metrics reflect state; `xml_format_reward` penalizes malformed output.
- `tests/verifiers_integration/test_end_to_end.py` — end-to-end rollout against a mock OpenAI client (canned responses), verifying trajectory structure, reward accumulation, and termination.

## Docker + Singularity

### `Dockerfile` (fully replaces the existing one)

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
  && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
  && mv /root/.local/bin/uv /usr/local/bin/uv \
  && mv /root/.local/bin/uvx /usr/local/bin/uvx

# Python 3.12
RUN uv python install 3.12

WORKDIR /opt/glyphbench

# Install deps first for cache
COPY pyproject.toml uv.lock /opt/glyphbench/
RUN uv sync --frozen --no-install-project

# Install source (editable)
COPY . /opt/glyphbench
RUN uv sync --frozen

# prime-rl + verifiers (locked via pyproject extras)
# (these come in via pyproject.toml [project.optional-dependencies] — see below)

ENV PATH="/opt/glyphbench/.venv/bin:${PATH}"

WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]
```

`pyproject.toml` changes:

- **Removed:** `gymnasium>=1.0`.
- **Core deps add:** `verifiers>=0.1.12`, `datasets>=3.0`, `openai>=1.50` (verifiers depends on it), `pydantic>=2.9` (kept).
- **New optional-extra `[rl]`:** `prime-rl` + `torch` + `flash-attn` (heavy; only needed for training containers).
- **`[eval]`** extra: `vllm>=0.8` (kept; still used for local inference server).

### `scripts/build_sif.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
IMAGE=${IMAGE:-glyphbench:latest}
SIF=${SIF:-glyphbench.sif}
docker build -t "$IMAGE" .
apptainer build --force "$SIF" "docker-daemon://$IMAGE"
echo "SIF ready at $SIF"
```

Explicit cleanup of the prior SIF happens via `--force`. The user's earlier SIF (`cluster_manager/glyphbench.sif`, 8.87 GB) is overwritten on rebuild.

### Container smoke test

After build, within the container:

```bash
uv run python -c "import glyphbench; print(len(glyphbench.all_glyphbench_env_ids()))"
# expect: 292

# start local vLLM server (background)
uv run vllm serve Qwen/Qwen3-0.6B --port 8000 &
sleep 60

uv run vf-eval glyphbench -m Qwen/Qwen3-0.6B -b http://localhost:8000/v1 \
  -x '{"env_id":"glyphbench/dummy-passthrough-v0","num_episodes":2}' -t 512
# expect: completes, writes results
```

## prime-rl wiring

### Eval config (`configs/eval/glyphbench-debug.toml`)

```toml
# Minimal smoke config for prime-rl eval entrypoint
model = "Qwen/Qwen3-0.6B"
environments = ["glyphbench"]
extra_env_kwargs = { env_id = ["glyphbench/dummy-passthrough-v0", "glyphbench/minigrid-empty-5x5-v0"], num_episodes = 2 }
max_tokens = 512
temperature = 0.7
sampling_args = { top_p = 0.9 }

[inference]
base_url = "http://localhost:8000/v1"
```

### RL smoke config (`configs/rl/glyphbench-smoke/`)

- `train.toml`: FSDP trainer targeting `Qwen/Qwen3-0.6B`, 1 GPU, LoRA off, 20 orchestrator steps, `micro_batch_size=1`, GRPO objective, low LR.
- `orch.toml`: points at `glyphbench` env with `env_id="glyphbench/minigrid-empty-5x5-v0"`, `num_episodes=4`, `n_frames=4`, `max_turns=50`.
- `infer.toml`: vLLM serving Qwen3-0.6B on `localhost:8001`, `max_model_len=4096` (more than enough for 512 output × 4-frame history × ~4k context).

Smoke condition: `uv run rl --trainer @ train.toml --orchestrator @ orch.toml --inference @ infer.toml` runs for ≥ 1 full orchestrator step end-to-end without crashing. **Not** tuned for quality — just proves the pipeline works against the new env package.

## Error handling

- **Parse failure** → parser returns noop action + `parse_failed=True`; `state["parse_failures"] += 1`. Episode continues.
- **`max_turns` reached** → env returns `truncated=True`; `is_done` fires; episode ends cleanly.
- **Unknown env_id passed to `load_environment`** → `KeyError` with list of known ids.
- **Game raises during `_step`** → wrapped as `vf.Error`, stored in `state["error"]`, has_error stop condition fires. Rubric sees partial trajectory.
- **Model emits no `<action>` tag** → noop, parse failure metric bumped.
- **vLLM server unreachable** → verifiers raises `vf.ModelError`, rollout ends in error state.
- **Empty frame history (turn 0)** → `[History]` section omitted entirely; no spurious `(no history yet)` placeholder.

## Execution plan (high-level, details go in the plan doc)

Six milestones, suite-by-suite:

1. **M1 Core + wrapper.** Rename ascii→glyph, drop `gym.Env` inheritance, introduce registry, build verifiers wrapper (env + parser + prompting + rubric), delete `harness/` / `providers/` / `runner/`, port `dummy/` suite, update pyproject, regenerate `uv.lock`. Tests for core and wrapper pass.
2. **M2 Minigrid (71 envs).** Swap imports + registration; update any test that relied on gym wrappers. Tests pass.
3. **M3 Classics (50) + Procgen (16).**
4. **M4 Atari (57) + Minihack (63) + Craftax (35).** All 292 envs now running non-gym.
5. **M5 Scripts + eval runner + random baseline.** Port demo/play/replay/record_random_gifs scripts to non-gym API. Regenerate `eval/random_baseline.json` under the new harness (5-episode quick run just to have a file; full baseline can be regenerated separately). Rewrite `eval/README.md` for vf-eval.
6. **M6 Container + prime-rl smoke.** New Dockerfile, build.sh, build SIF. prime-rl debug eval config + RL smoke config. Local smoke tests:
    - `uv run vf-eval glyphbench -m Qwen/Qwen3-0.6B -x '{"env_id":"glyphbench/dummy-passthrough-v0","num_episodes":2}' -t 512` → passes.
    - `uv run eval @ configs/eval/glyphbench-debug.toml` → passes against local vLLM.
    - `uv run rl ...smoke config...` → runs ≥ 1 orchestrator step without error.
    - SIF built via `scripts/build_sif.sh` and the same `vf-eval` command works inside `apptainer run glyphbench.sif …`.

All six milestones happen in one session via subagent-driven-development (dispatched task-parallel where safe; core-first serialized).

## Verification checklist before declaring done

- `grep -r "gymnasium\|import gym\b\|gym\.Env\|gym\.register\|gym\.make" src/ tests/ scripts/ eval/` → returns nothing.
- `grep -r "ascii" src/ tests/` → only references to literal "ASCII" in game narrative text (if any), no code/class/module names.
- `uv run python -c "import glyphbench; assert len(glyphbench.all_glyphbench_env_ids()) == 292"` succeeds.
- `uv run pytest` runs all tests green.
- `uv run vf-eval glyphbench -m Qwen/Qwen3-0.6B -x '{"env_id":"glyphbench/dummy-passthrough-v0","num_episodes":2}' -t 512` completes with non-crashed rollouts.
- `uv run eval @ configs/eval/glyphbench-debug.toml` completes against a running Qwen3-0.6B vLLM server.
- `uv run rl --trainer @ configs/rl/glyphbench-smoke/train.toml --orchestrator @ configs/rl/glyphbench-smoke/orch.toml --inference @ configs/rl/glyphbench-smoke/infer.toml` completes ≥ 1 orchestrator step.
- `bash scripts/build_sif.sh` produces `glyphbench.sif`; running the same `vf-eval` inside `apptainer run` succeeds.
