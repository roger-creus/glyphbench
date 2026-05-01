# Observation format and harness

## What the agent sees per turn

The harness shows the model a single text string composed of these blocks:

- `[Legend]` — glyph → meaning mapping (rendered once, deduped per rollout).
- `[Grid]` — the 2D Unicode grid (the only required channel).
- `[HUD]` — optional complementary state that cannot be read from the grid
  (turn budget, HP, score, inventory, velocity, cooldowns, etc.).
- `[Message]` — optional per-turn narrative event ("You bumped a wall.").
- `[Actions]` — the action vocabulary the model must pick from this turn.

The grid remains authoritative for spatial state. The HUD must not repeat
positions, visible entity locations, or facing already encoded by directional
glyphs; it is only for state the Unicode grid cannot convey.

## Single-codepoint Unicode glyphs

Every cell is exactly one codepoint, with no symbol collisions inside a
suite. Common glyphs:

| Glyph | Meaning |
|---|---|
| `█` | Wall |
| `→ ↓ ← ↑` | Player direction |
| `★` | Goal |
| `≈` | Water |
| `▣` | Door |
| `·` | Floor |

(Each suite uses its own additional glyphs; see the per-env `[Legend]` for the
full mapping.)

## System prompt

`env.system_prompt()` returns a compact rules / actions / reward / termination
description, ready to pass as a system message to any LLM. The system prompt
also advertises the output-token budget the eval is run with so the model
can self-pace its reasoning.

## Frame-stacked history

`load_environment(..., n_frames=N)` joins the last N observation strings into
a single user message per turn. Default: `n_frames=0` (stateless / pure
Markov).

## Memory mode

Memory mode is the default. Each environment step uses two model generations:

1. **Action** — same as before. Prompt is `[system, user_obs_t]`; completion
   is `<think>...</think><action>NAME</action>`.
2. **Memory update** — prompt is `[system, user_obs_t, assistant_action_t,
   lean_memory_user]`; completion is `<memory>...</memory>` (thinking is
   forced off via `chat_template_kwargs.enable_thinking=False`). The lean
   memory_user contains only the env's reaction to the last action: reward,
   `terminated`, `truncated`, plus the write instruction. Previous memory
   and the action chosen are visible via the conversation prefix; the
   action's `<think>` content is stripped by the chat template the same way
   it is for any prior assistant turn.

The two generations are stored as **two separate trajectory steps** in the
verifiers `results.jsonl` — an `action` step followed by a `memory` step —
each with its own `prompt`/`completion` pair. The split is required because
prime-rl's `pretokenize_rollout_trajectory` rejects trajectory steps whose
completion contains mixed roles. Storing them separately guarantees every
step's completion is purely assistant tokens, and both steps share the same
per-turn task reward so action and memory tokens are both eligible for
on-policy RL.

`memory_update_max_tokens` (default 4096) caps only the second generation's
output. The memory text persists across turns: the next user observation
opens with a `[Memory]` block carrying the stored text, with the current
grid still authoritative on conflicts. Pass `use_memory=False` for the
single-generation loop. Memory-aware trajectories show previous and updated
memory in `gb replay`; the standalone trajectory / GIF renderer also
includes stored memory when present.

### Failure handling

Both failure modes are surfaced explicitly:

- **Memory parse failure** (no `<memory>...</memory>` tag in the response):
  the trajectory step's extras carry `memory_parse_failed=True`, the
  rollout-level counter `state["memory_parse_failures"]` is incremented,
  and the rubric reports `memory_parse_failure_rate`. The previous memory
  is **retained** so the next turn's `[Memory]` block still shows the
  most recent valid scratchpad.

- **Memory output truncation** (response hit `memory_update_max_tokens`):
  the trajectory step's `is_truncated=True`, the counter
  `state["memory_completion_truncations"]` is incremented, and the rubric
  reports `memory_completion_truncation_rate`. The truncation is
  surfaced *to the next memory turn* via the `Output truncated` line in
  `[Last Action]`, so the model sees that the prior memory write was cut
  off and can choose to be more concise.

### Budget arithmetic (`max-model-len`)

The memory call's input is the action call's input plus the
`assistant_action_T` tag, the re-injected action reasoning, the
[Next Observation] block, and the [Memory Update] instruction. With
`max_output_tokens=8192` (action) and `memory_update_max_tokens=4096`
(memory) the worst-case total is roughly:

```
prompt (sys+obs)         ~  8000
+ assistant_action tag   ~    50
+ [Last Action] reasoning≤  8192
+ [Env Response]         ~    50
+ [Next Observation]     ~  1500
+ [Memory Update]        ~   200
+ memory output budget   =  4096
                         = ~22000  → fits comfortably in 32768
```

`max-model-len=32768` is the recommended floor when memory mode is on
(matches the cluster default in `cluster_manager/config.py`). The local
`eval/run_full.sh` and `run_debug.sh` use 32768. With long-history runs
(`n_frames>0`) or unusually large grids you may want to bump higher.

## Determinism

All observations are deterministic — identical seeds produce identical
trajectories. There is no privileged side-channel to the agent: every
game-relevant fact must be readable off the visible glyphs.

## Full `load_environment` signature

```python
glyphbench.load_environment(
    task_id: str | list[str] | None = None,
    num_episodes: int = 5,
    n_frames: int = 0,
    max_turns: int | None = None,
    max_output_tokens: int = 512,
    seed: int = 42,
    use_memory: bool = True,
    memory_update_max_tokens: int | None = 4096,
) -> verifiers.Environment
```

The returned `verifiers.Environment` (concretely `GlyphbenchMultiTurnEnv`) is
ready to plug into `prime eval run` or any RL training loop that consumes
verifiers envs.
