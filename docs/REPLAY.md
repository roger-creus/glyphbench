# `glyphbench replay`

A rich TUI for inspecting saved rollouts. Renders the per-turn
ASCII grid alongside the agent's reasoning, action, memory state,
and any per-turn parse errors.

## Quick start

```bash
# List every (model, env, seed) tuple under a runs dir
uv run glyphbench replay cluster_manager/results --list

# Play every rollout that matches one env (filters AND-combine)
uv run glyphbench replay cluster_manager/results \
    --env glyphbench/minigrid-empty-6x6-v0

# Step through a single rollout turn-by-turn
uv run glyphbench replay cluster_manager/results \
    --env glyphbench/craftax-fight-cow-v0 --pause

# Pace continuous playback (default 0.15s per turn)
uv run glyphbench replay runs/local-eval --delay 0.4
```

`runs_dir` may point at the cluster_manager results tree, a single
per-run-hash subdirectory (e.g.
`cluster_manager/results/evals/glyphbench--Qwen--Qwen3.5-4B/<hash>`),
or any directory tree containing `results.jsonl` files.

## CLI flags

| Flag | Effect |
|---|---|
| `--env`, `--suite`, `--model`, `--seed` | Repeatable filters; AND-combined |
| `--episode N` | 0-indexed pick from the filtered set; plays only that one |
| `--list` | Print the matched index and exit (no playback) |
| `--pause` | Step turn-by-turn instead of timed playback |
| `--delay N` | Seconds between turns in continuous mode (default 0.15) |

When stdout isn't a TTY (piped, redirected) the renderer auto-falls
back to a plain grid + memory dump.

## Pause-mode keys

In `--pause` mode the playback is fully random-access:

| Key | Action |
|---|---|
| **→** / any other key | next turn |
| **←** | previous turn |
| **q** | exit this rollout, advance to the next match |
| **s** | open full system prompt in `$PAGER` |
| **r** | open full reasoning chain for the current turn in `$PAGER` |
| **l** | open full legend (glyph table) for the current turn in `$PAGER` |
| **m** | open previous + updated memory (side-by-side) for the current turn in `$PAGER` |

`$PAGER` defaults to `less -R`. Inside the pager use the usual less
keys (`q` to return, `/` to search, `g`/`G` for top/bottom). The
terminal is reset to cooked mode automatically when you exit the
pager AND on every replay-tool exit path, so the shell never gets
left with echo disabled.

## Panel layout

```
+---------------------- HEADER BAR -------------------------+
|  model · env_id            turn N/T   seed   reward   ... |
|  [optional warning chips: PARSE FAIL, parse-fallback, ...] |
+---------------------- SYSTEM PROMPT ----------------------+
|  (clipped; press `s` in pause mode to scroll)             |
+---------------- BODY ------------------------+------------+
| LEFT (ratio 2)               | RIGHT (ratio 3)             |
|  +---- grid ----+            |  +-- turn errors --+ (red)  |
|  |  ASCII grid  |            |  +- previous memory +       |
|  |  (auto-sized)|            |  +-- step Step:T/N+ (yellow)|
|  +--------------+            |  +-- HUD ---------+         |
|                              |  +-- legend ------+         |
|  +--- reasoning ----+        |  +-- action ------+         |
|  |  fills the rest  |        |  +-- env feedback+         |
|  |  of the column   |        |  +-- updated memory+       |
|  +-------------------+       |                             |
+------------------------------+----------------------------+
```

Notes on the layout:

- The grid panel is sized to its content height — empty space below
  it is reclaimed by the **reasoning** panel, which is the largest
  potential consumer.
- The `step` indicator (`Step: T / N`) is split out of the HUD into
  its own dedicated yellow panel.
- `previous memory` and `updated memory` only appear when the
  rollout was produced with `use_memory=True`.
- The `turn errors` panel only appears when the current turn had a
  detectable parse / format issue. The grid border ALSO turns red
  when this happens, and the `action` / `reasoning` panel borders
  flip red on their respective failures.
- Long content (system prompt, reasoning, memory, legend) is
  pre-clipped to fit the panel; press the matching pager hotkey in
  pause mode to view the full text scrollably.

## How the parser stays aligned with eval

Action extraction goes through `GlyphbenchXMLParser._extract_candidate`
— the same parser the verifiers eval scores against. When the env
can be instantiated (cached via `_spec_for_env_id`), the displayed
action is canonicalised through `parser.parse_action(text, spec,
noop=...)` so the panel shows the exact name the eval recorded
(including the env's noop on parse failure). Without spec, the
candidate is shown raw with a small CLI-side cleanup pass for
malformed forms like `ACTION_NAME=MOVE_FORWARD`.

Reasoning extraction handles Qwen3.5's chat template prefill — the
template inserts `<think>\n` on the model's behalf so the stored
assistant content typically has only `</think>`, never the opener.
The CLI treats start-of-string as an implicit opener when only the
closer is present.
