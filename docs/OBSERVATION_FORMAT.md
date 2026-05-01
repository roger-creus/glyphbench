# Observation format and harness

## What the agent sees per turn

The harness shows the model a single text string composed of these blocks:

- `[Legend]` — glyph → meaning mapping (rendered once, deduped per rollout).
- `[Grid]` — the 2D Unicode grid (the only required channel).
- `[Message]` — optional per-turn narrative event ("You bumped a wall.").
- `[Actions]` — the action vocabulary the model must pick from this turn.

Envs may compute a `[HUD]` (HP, inventory, score, etc.) for their `info`
dict and trajectory logs, but the harness deliberately does **not** show it
to the model. Privileged state (mob positions, hidden inventory) must
therefore be encoded in the visible grid for the agent to reason about.

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
