# atlas_rl.harness

The agent harness is Markovian: every turn's prompt is fully determined by
`(env.system_prompt, EpisodeState, current_observation)` — no observation history
and no accumulated plan history. The LLM's persistent reasoning lives in the
`strategic_plan`, `tactical_plan`, `subgoals`, and `lessons` fields of
`EpisodeState`, which the LLM rewrites each turn through its structured JSON output.

## Public types

- `EpisodeState` — persistent episode state (plan, subgoals, lessons, recent_actions).
- `HarnessOutput` — pydantic model for the LLM's per-turn JSON output.
- `HarnessAgent` — the orchestrator.

## Invariants

1. `recent_actions` has `maxlen=5`, hardcoded.
2. `lessons` are append-only within an episode (LLM can't retract).
3. `subgoals` can be rewritten each turn (add/mark_done).
4. `thinking` field is discarded after the turn; never persisted, never re-shown.
5. Parse retries are ≤ 3; after that, fall back to the env's `noop_action_name`.
6. Malformed output produces `TurnMetrics.action_parse_error=True` and counts parse retries.

See `specs/2026-04-12-stage-0-foundation-and-pilot.md` §4 for the full contract.
