# LLM Agent Failure Modes — Glossary & Diagnostic Guide

This document is the canonical reference for every failure-mode label glyphbench
emits when running LLM agents (eval or RL training). Each metric name is
distinct and unambiguous so you never have to ask "which kind of truncation?".

## Terminology

| Term | Meaning |
|---|---|
| **Episode terminated** | The env reached a terminal state — typically goal reached, agent died, or game-over condition. Distinct from "truncated". |
| **Episode truncated (max_turns)** | The env's per-rollout `max_turns` budget hit before any terminal condition fired. Means "ran out of clock", not "agent failed". |
| **Action turn** | The first of two LLM calls per env step in memory mode (or the only call without memory mode). Generates `<think>...</think><action>NAME</action>`. |
| **Memory turn** | The second LLM call per env step (memory mode only). Generates `<memory>...</memory>`. `enable_thinking=False`. |
| **Action completion truncated** | The action turn's response hit `max_tokens` (8192 today) before emitting a closing `</action>` tag. |
| **Memory completion truncated** | The memory turn's response hit `max_tokens` (4096 today) before closing `</memory>`. |
| **Forfeit** | The action turn's response could not be parsed (no `<action>` tag, or unknown NAME), so the env was not stepped: state unchanged, turn counter advances, reward 0, action_chosen='FORFEIT'. |
| **Memory parse failure** | The memory turn emitted no valid `<memory>...</memory>` tag, so the previous memory is retained. The env has already been stepped at this point — no impact on rewards. |
| **Prompt overlong** | The input exceeded the server's context cap (16384 input tokens in our config). vLLM rejects the call; the rollout crashes loudly. By design this should never fire. |

## Per-step extras (`TrajectoryStep.extras`)

| Key | Type | On steps | Meaning |
|---|---|---|---|
| `glyphbench_step_role` | `"action"` / `"memory"` | both | Which of the two memory-mode calls this step represents. Non-memory rollouts have only "action". |
| `parse_failed` | bool | action | True when the action couldn't be parsed (`<action>` missing or NAME unknown). |
| `parse_failure_reason` | str / None | action | `"no_action_tag"` (no complete `<action>...</action>`) or `"unknown_name"` (NAME not in spec). `None` on success. |
| `action_chosen` | str | action | Action applied to the env, OR `"FORFEIT"` when `parse_failed=True`. |
| `forfeit` | bool | action | True iff `parse_failed=True`. Mirror for filtering. |
| `memory_parse_failed` | bool | memory | True when no `<memory>` tag was emitted; previous memory was retained. |
| `stored_memory` | str | memory | The memory string in effect after this step. |

`TrajectoryStep.is_truncated` (verifiers' built-in) flags completion truncation
on either step type.

## Per-episode counts (`state[...]`)

| Key | Meaning |
|---|---|
| `forfeit_count` | Action turns with `forfeit=True`. |
| `action_completion_truncations` | Action turns with `is_truncated=True`. |
| `memory_completion_truncations` | Memory turns with `is_truncated=True`. |
| `memory_parse_failures` | Memory turns with `memory_parse_failed=True`. |
| `num_action_turns` | Total action turns this rollout. |
| `num_memory_turns` | Total memory turns this rollout (memory mode only). |
| `num_turns` | Env-action turns (counts forfeits as turns; equals `num_action_turns`). |
| `terminated` | True if env reached a terminal state. |
| `truncated` | True if env's `max_turns` budget was hit. |
| `episode_return` | Sum of per-turn rewards. |

## Rubric metrics (rates)

All rates are floats in [0, 1] aggregated per rollout (not per-step).

| Metric | Formula | Diagnostic guidance |
|---|---|---|
| `episodic_return` (weight=1.0, the trained signal) | `sum(rewards)` over turns | The optimisation target. |
| `episode_length` | `num_turns` | Higher = longer episodes. |
| `episode_terminated_rate` | `1 if terminated else 0` | Across rollouts: fraction that hit a terminal state. |
| `episode_truncated_max_turns_rate` | `1 if truncated else 0` | Fraction that ran out of env clock. High value alongside low return → agent is too slow / getting lost. |
| `forfeit_rate` | `forfeit_count / num_turns` | High value → model is failing to emit valid `<action>` tags. Often correlates with action-completion truncation; check both. |
| `action_completion_truncation_rate` | `action_completion_truncations / num_action_turns` | High value → reasoning chains exceed the 8192-token output cap. Tighten system prompt, raise the cap, or use a less verbose model. |
| `memory_completion_truncation_rate` | `memory_completion_truncations / num_memory_turns` | High value → memory writes are exceeding 4096 tokens. Trim the memory schema or raise the cap. |
| `memory_parse_failure_rate` | `memory_parse_failures / num_memory_turns` | High value → model is emitting memory without `<memory>` tags despite `enable_thinking=False`. Check chat template. |
| `xml_format_reward` | verifiers' built-in | Format compliance signal. |

### Diagnostic flowchart (informal)

- `forfeit_rate` high ⟶ check `action_completion_truncation_rate`. If also high, root cause is "reasoning too long". If low, root cause is "model not following format" — strengthen the system-prompt format example.
- `episode_truncated_max_turns_rate` high alongside low `episodic_return` ⟶ agent is wandering / not making progress. Look at trajectories.
- `memory_parse_failure_rate` high ⟶ chat template likely leaking `<think>` despite `enable_thinking=False`, OR memory mode prompt isn't conditioning the model strongly enough. Check `MEMORY_BLOCK_TMPL` in `prompting.py`.

## Replay rendering (`gb replay`)

Per-turn header chips:

| Chip | Means |
|---|---|
| `[forfeit]` | Action turn forfeited — env was not stepped. |
| `[trunc-action]` | Action completion was truncated. |
| `[trunc-memory]` | Memory completion was truncated (memory mode only). |
| `[mem-parse-fail]` | Memory turn emitted no `<memory>` tag; previous memory retained. |

## Plotting

`plotting/common.py` consumes the rate metrics above directly. Updates to the
plotting suite live in `paper_figures/` and `notebooks/`; both ingest metric
names verbatim, so renames here automatically propagate.

## Implementation pointers

- Strict parser: `src/glyphbench/verifiers_integration/parser.py`
- Forfeit semantics: `BaseGlyphEnv.forfeit_turn` in `src/glyphbench/core/base_env.py`
- Memory turn redesign: `src/glyphbench/verifiers_integration/memory.py`
- Per-step extras + state plumbing: `src/glyphbench/verifiers_integration/env.py`
- Rubric metrics: `src/glyphbench/verifiers_integration/rubric.py`
- Replay rendering: `src/glyphbench/cli.py`
