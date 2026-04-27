# GlyphBench Memory Scaffold Design

Date: 2026-04-27

## Goal

Add an opt-in memory scaffold to GlyphBench's `verifiers` integration that works across every registered environment without changing individual game implementations.

When enabled, each environment step uses two model generations:

1. an action generation that sees the previous memory plus the current observation, and
2. a memory-update generation that sees the full action-selection exchange, environment feedback, and next observation.

These two generations together remain one environment turn for episode length, rewards, metrics, saved rollouts, visualization, and RL training.

## Non-Goals

- Do not add memory to the direct `BaseGlyphEnv` API or individual env classes.
- Do not require a new upstream `verifiers` release.
- Do not make memory mode the default evaluation protocol.
- Do not train a separate auxiliary memory reward.
- Do not mention exact memory token budgets inside the model-facing prompts.

## Public API

Extend `glyphbench.load_environment(...)` with opt-in memory arguments:

```python
load_environment(
    ...,
    use_memory: bool = False,
    memory_update_max_tokens: int | None = None,
)
```

`use_memory=False` preserves the current single-generation-per-turn behavior.

`memory_update_max_tokens=None` means the memory-update generation reuses the action turn's sampling token limit. If a caller supplies a value, only the memory-update generation uses that override.

The existing `max_output_tokens` argument remains the advertised action-response budget in the system prompt. It does not become a model-visible memory budget.

## Prompt Protocol

### Action Prompt

When memory is enabled, `render_user_turn(...)` receives the current stored memory and inserts a `[Memory]` block before the observation:

```text
[Memory]
Use this as carried state from previous turns. The current observation is authoritative if it conflicts.

<memory>
...
</memory>

[Legend]
...

[Current Observation - turn T]
...

[Actions]
Choose one: [...]

Now emit your move as `<action>ACTION_NAME</action>`.
```

The action response contract stays compatible with current GlyphBench:

```text
<think>optional reasoning</think>
<action>ACTION_NAME</action>
```

Only `<action>` is required for action parsing.

### Memory-Update Prompt

After parsing the action and stepping the environment, GlyphBench prompts the same model to update memory. This prompt is conditioned on:

- previous memory,
- the full action prompt,
- the full action assistant response, including any action-selection thinking,
- parsed action name,
- reward from the just-applied action,
- terminated/truncated flags,
- next observation.

The model-facing instruction asks it to keep memory concise, but does not mention exact token limits.

Example shape:

```text
[Memory Update]
Update the memory that will be shown on the next environment turn. Keep it concise.

[Previous Memory]
<memory>
...
</memory>

[Action Response]
...full assistant action response...

[Environment Feedback]
Parsed action: EAST
Reward: +0.000
Terminated: false
Truncated: false

[Next Observation]
...

Write the updated memory.
```

The memory-update assistant may emit thinking and may optionally use memory tags:

```text
<think>optional reasoning</think>
remembered facts here
```

or:

```text
<think>optional reasoning</think>
<memory>
remembered facts here
</memory>
```

## Memory Extraction

Only the memory-update assistant output determines the memory stored for the next environment turn.

Extraction order:

1. If one or more `<memory>...</memory>` blocks are present, concatenate their contents with blank lines and store that text.
2. Otherwise, if `</think>` appears, store the text after the final `</think>`.
3. Otherwise, strip any think tags and store the remaining assistant text.

Stored memory excludes the memory-update thinking. Raw memory-update responses remain available in trajectory metadata for audit and replay.

## Memory Length Control

Memory-update generation length is controlled at model-call time, not by the action prompt. By default, the memory update uses the same generation limit as the action call. `memory_update_max_tokens` can override it.

There is no separate memory-only token limit after extraction. If the memory-update generation hits its completion limit, the harness stores whatever memory can be extracted from the generated text.

The model-facing prompt says only to keep the memory concise.

## RL Training Semantics

Memory mode must produce one trainable trajectory step per environment step, not two independent training turns.

For each environment step, the combined `TrajectoryStep` has:

- `prompt`: the action prompt messages.
- `completion`: the action assistant message, the memory-update user message, and the memory-update assistant message.
- `tokens.prompt_ids`: token IDs for the action prompt.
- `tokens.prompt_mask`: zeros for the action prompt.
- `tokens.completion_ids`: token IDs for the action assistant plus memory-update user prompt plus memory-update assistant.
- `tokens.completion_mask`: ones for action assistant tokens, zeros for memory-update user-prompt tokens, ones for memory-update assistant tokens.
- `tokens.completion_logprobs`: logprobs for assistant-generated tokens and zeros for memory-update user-prompt bridge tokens.

The trainer still sees the memory-update user prompt as conditioning context. It is masked only for loss.

Both trainable assistant regions receive the same task-level reward and advantage as the usual action response. There is no separate memory reward.

## Token Stitching Strategy

The pinned `verifiers` token client can stitch later prompts from previous trajectory steps using exact engine tokens. The memory implementation should preserve this property without patching upstream `verifiers`.

Runtime strategy:

1. Generate the action response through the normal `get_model_response(...)` path.
2. Build a temporary action-only `TrajectoryStep` and append it to `state["trajectory"]`.
3. Build the memory-update prompt as the previous conversation plus one new user message.
4. Generate the memory-update response. The `verifiers` token client can now stitch this second prompt from the temporary action step.
5. Merge token data from the action response and memory-update response into one combined `TrajectoryStep`.
6. Replace the temporary action-only step with the combined step.

If either response lacks token data, the combined step should still be recorded for evaluation and replay, but `tokens` may be `None`, matching existing behavior for non-token clients.

## Environment State

Memory state lives in the rollout `state`, not in the game object:

- `state["memory"]`: stored memory text shown on the next action prompt.
- `state["memory_enabled"]`: boolean feature flag.
- `state["frames"]`, `state["current_obs"]`, and game state remain the existing source of environment observations.

Resetting a rollout starts with empty memory.

The `turn` counter and `max_turns` continue to count environment actions only. Memory-update generations do not increment `game.turn`.

## Saved Trajectory Metadata

Each memory-enabled combined `TrajectoryStep["extras"]` should include:

```python
{
    "glyphbench_memory": {
        "enabled": True,
        "previous_memory": str,
        "action_prompt": list[dict],
        "action_response": list[dict],
        "memory_update_prompt": list[dict],
        "memory_update_response": list[dict],
        "parsed_memory": str,
        "stored_memory": str,
        "extraction_mode": "tag" | "post_think" | "stripped_text",
        "memory_update_was_truncated": bool,
    }
}
```

The exact message objects should use the same serializable message shape already written in `results.jsonl`.

## Visualization

`glyphbench replay` must recognize memory metadata and render memory-enabled rollouts as one frame per environment step.

For memory-enabled rollouts, rich replay should group:

- grid/current observation,
- previous memory,
- action reasoning and parsed action,
- reward/status feedback,
- updated stored memory,
- optional raw memory-update response when useful for debugging parse failures.

Plain replay remains grid-first. It can include concise memory/update text when memory metadata exists, but must preserve a simple grid-only path for terminals and scripts.

Existing non-memory rollouts must render exactly as before.

The older `scripts/replay_trajectory.py` and `scripts/record_random_gifs.py` operate on a random-agent JSONL schema rather than model `results.jsonl`; they do not need memory support for this feature.

## Error Handling

- If the action response cannot be parsed, GlyphBench uses the current parser fallback and still asks for a memory update using the applied fallback action and feedback.
- If the memory-update response is empty or extraction produces empty text, the next memory becomes empty unless the implementation explicitly preserves previous memory. The first implementation should use the extracted result directly, even if empty, because the model is responsible for rewriting memory.
- If the memory-update generation fails after the action was already applied, record the action step, mark the rollout error consistently with existing `verifiers` behavior, and avoid silently inventing memory.
- If token merging fails, keep the rollout evaluable and replayable with `tokens=None` rather than corrupting masks.

## Testing Plan

Unit tests:

- `load_environment(..., use_memory=True)` stores memory settings and leaves default behavior unchanged when disabled.
- Initial action prompt includes an empty `[Memory]` block when memory is enabled.
- Later action prompt includes the memory parsed from the prior memory update.
- Memory extraction handles `<memory>` tags, post-`</think>` text, and stripped text fallback.
- Memory-update prompt includes full action response, parsed action, reward, done flags, and next observation.
- One environment step produces one trajectory item in memory mode.
- Combined completion masks train action assistant tokens and memory-update assistant tokens while masking memory-update user prompt tokens.
- Empty or malformed action responses still produce a memory-update prompt after fallback action application.

Visualization tests:

- `_build_turns` or its replacement groups memory-enabled trajectory steps by `extras["glyphbench_memory"]`.
- Rich replay receives previous and updated memory for memory-enabled rollouts.
- Non-memory replay output grouping stays unchanged.

Integration tests:

- A dummy-env rollout with scripted action and memory-update responses reaches the goal and records memory in trajectory extras.
- A token-level synthetic merge test verifies combined IDs, masks, and logprobs without requiring a live model server.

Validation commands:

```bash
uv run pytest tests/verifiers_integration -q
uv run pytest tests/test_cli.py -q
uv run ruff check src/glyphbench/verifiers_integration src/glyphbench/cli.py tests/verifiers_integration tests/test_cli.py
```

If full dependency installation is unavailable locally, run the focused tests that do not require heavy extras and report any skipped verification explicitly.

## Resolved Decisions

- Memory mode is opt-in.
- Full RL support is required in the first implementation.
- Memory-update assistant tokens are trained end to end with task rewards.
- Memory-update user prompt tokens are visible conditioning context but masked for loss.
- Memory update is conditioned on environment feedback and next observation.
- Memory update is also conditioned on full action-selection thinking.
- Model-facing prompt says keep memory concise and does not expose exact token budgets.
- If `<memory>` tags exist, parse tag contents. Otherwise use post-think text or stripped assistant text.
