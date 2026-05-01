# Use GlyphBench with your own agent

## Direct game loop

```python
import glyphbench
from glyphbench.core import make_env

env = make_env("glyphbench/minigrid-doorkey-6x6-v0")
obs, info = env.reset(42)

done, total = False, 0.0
while not done:
    action = your_agent(obs, env.action_spec.names)
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    done = terminated or truncated

print(f"Episode return: {total}")
env.close()
```

`your_agent(obs, action_names)` receives:

- `obs` — a single text string with `[Legend]`, `[Grid]`, optional `[Message]`,
  and `[Actions]` blocks. See `docs/OBSERVATION_FORMAT.md`.
- `action_names` — the list of action names valid this turn.

It must return either an integer action index or one of the action name strings
— `env.step` accepts both.

`make_env` returns a `BaseGlyphEnv` with the following interface:

| Attribute / method | Description |
|---|---|
| `env.reset(seed)` | Reset the episode; returns `(obs, info)`. |
| `env.step(action)` | Advance one turn; returns `(obs, reward, terminated, truncated, info)`. |
| `env.action_spec.names` | `list[str]` — canonical name per action index. |
| `env.action_spec.n` | `int` — discrete action count. |
| `env.action_spec.render_for_prompt()` | LLM-facing action menu string embedded in the system prompt. |
| `env.system_prompt()` | Full system prompt string for this env (rules + action table). |
| `env.close()` | Release resources. |

## Loading as a verifiers env (for evaluation or RL)

```python
import glyphbench

vf_env = glyphbench.load_environment(
    task_id="glyphbench/minigrid-empty-5x5-v0",
    num_episodes=5,
    n_frames=0,
    max_output_tokens=8192,  # match the budget your agent runs at
)
# Memory mode is on by default (two-generation turn: action then
# <memory>...</memory> write). Pass use_memory=False to disable.
```

Full signature:

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

- `task_id=None` samples uniformly across all registered envs.
- `task_id=list[str]` interleaves multiple envs within the returned environment.
- `n_frames` controls how many past grid frames are appended to the observation
  for context (0 = current turn only).
- `use_memory` enables a persistent scratchpad updated between turns. Default
  is `True` — every env turn becomes an action generation followed by a
  `<memory>...</memory>` write, with a 4096-token budget on the memory call.
  Pass `use_memory=False` for the legacy single-generation loop.

Returns a `verifiers.Environment` (concretely `GlyphbenchMultiTurnEnv`). Plug
into `prime eval run` or any RL trainer that consumes verifiers envs. See
`eval/README.md` for the canonical eval invocation.

## Inspecting trajectories

GlyphBench writes verifiers-format `results.jsonl` files (one row per
example, with full per-turn `prompt` / `completion` / `reward` / `info`).
Two tools read these:

### `gb replay` — rich TUI

```bash
uv run glyphbench replay <runs_dir> --env glyphbench/<env_id>
```

Multi-panel TUI: header bar / system prompt / grid / reasoning / memory / HUD
/ legend / action / env feedback. Pause-mode hotkeys (`s` system / `r`
reasoning / `m` memory / `←/→` step / `q` next) drop into a pager for any
panel. Full reference: `docs/REPLAY.md`.

### `scripts/replay_trajectory.py` — single-file viewer + GIF export

```bash
# Replay one .jsonl in terminal with color
uv run python scripts/replay_trajectory.py path/to/trajectory.jsonl

# Export it as a GIF
uv run python scripts/replay_trajectory.py path/to/trajectory.jsonl --gif out.gif

# Replay all .jsonl files in a directory
uv run python scripts/replay_trajectory.py path/to/trajectories/
```

The script renders Unicode glyphs with terminal color and can export
frame-by-frame GIFs for sharing or inspection outside a terminal.

## Determinism and seeding

All envs are deterministic given a seed. `env.reset(seed)` re-seeds the env's
RNG; subsequent `env.step(action)` calls are deterministic functions of
`(seed, action_sequence)`. This makes single-run rollouts trivially
reproducible — useful for regression tests and golden-file trajectories.

Passing the same `seed` to `load_environment` produces the same episode
ordering across runs.

## Action conventions

- Every env has `env.action_spec.names: list[str]` with the canonical name
  per action index.
- Every env has `env.action_spec.n: int`, the discrete action count.
- `env.action_spec.render_for_prompt()` returns the LLM-facing action menu
  string used in the system prompt — useful if you're building a custom
  prompt around our action vocabulary.
- Action names are `SHOUTY_SNAKE_CASE` by convention; the parser falls back
  to a simple regex extraction if JSON parsing fails.
- `env.step` accepts either an integer index or an action name string.
  Name matching is case-insensitive; cross-suite no-op aliases
  (`NOOP` / `WAIT` / `DONE` / `PASS` / `SKIP`) resolve automatically when
  the env defines one of them.
