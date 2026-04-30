# README public-release shine — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a compact, polished `README.md` plus per-area subdir docs and four HF-hosted GIF assets ahead of making the repo public.

**Architecture:** Throwaway tooling under `tools/readme-assets/` (never committed) renders all assets, then they're uploaded to the existing HF dataset and referenced from the README via raw URLs. Two rendering pipelines: a Pillow-based grid-only renderer (300 envs + hero) and asciinema+agg captures of the live TUI (replay + demo). A pre-rewrite README freshness audit drives the doc rewrites.

**Tech Stack:** Python 3.10+ · Pillow · ffmpeg · asciinema · agg · Rich (for TUI capture target) · `huggingface_hub` (existing `scripts/upload_assets.py`).

**Spec:** `docs/superpowers/specs/2026-04-30-readme-public-release-design.md`

**Important env switch from spec:** `gb_replay.gif` targets `glyphbench/minigrid-empty-5x5-v0` (not `minigrid-doorkey-6x6-v0`). Reason: `cluster_manager/results/evals/` already contains a saved trajectory for empty-5x5 but not for doorkey, and `gb replay` requires a saved JSONL. Smaller grid is also better for side-panel visibility.

---

## Task 1: Scaffold the throwaway workspace

**Files:**
- Create: `tools/readme-assets/` (local-only, never tracked)
- Create: `tools/readme-assets/out/gifs/` (local-only)
- Create: `tools/readme-assets/out/readme/` (local-only)

- [ ] **Step 1: Create the workspace directories**

```bash
mkdir -p /home/roger/Desktop/rl-world-ascii/tools/readme-assets/out/gifs
mkdir -p /home/roger/Desktop/rl-world-ascii/tools/readme-assets/out/readme
ls /home/roger/Desktop/rl-world-ascii/tools/readme-assets/out
```

Expected output: two directories printed (`gifs  readme`).

- [ ] **Step 2: Add a local-only ignore so we never accidentally `git add` it**

Append `tools/readme-assets/` to `.git/info/exclude` (local-only, not committed):

```bash
echo "tools/readme-assets/" >> /home/roger/Desktop/rl-world-ascii/.git/info/exclude
git -C /home/roger/Desktop/rl-world-ascii check-ignore -v tools/readme-assets/anything 2>&1
```

Expected output line: `.git/info/exclude:N:tools/readme-assets/  tools/readme-assets/anything` proving the ignore is active.

- [ ] **Step 3: No commit (no tracked changes)**

Verify with `git -C /home/roger/Desktop/rl-world-ascii status --short` — output should still only show `?? Craftax-main/`.

---

## Task 2: README freshness audit

**Files:**
- Create: `tools/readme-assets/audit.md` (local-only)
- Read: `README.md`, `pyproject.toml`, `src/glyphbench/core/__init__.py`, `src/glyphbench/__init__.py`, `src/glyphbench/cli.py`, `src/glyphbench/verifiers_integration/`, `eval/README.md`, `src/glyphbench/rl/README.md`, `scripts/`, `docs/REPLAY.md`, `CONTRIBUTING.md`, `eval/random_baseline.json`, `.env.cluster.template`, `configs/rl/qwen35-4b-glyphbench/README.md`

- [ ] **Step 1: Verify the per-suite env counts**

For each suite, count envs in the registry:

```bash
cd /home/roger/Desktop/rl-world-ascii && uv run python -c "
from glyphbench.core import all_glyphbench_env_ids
ids = [e for e in all_glyphbench_env_ids() if 'dummy' not in e]
from collections import Counter
suite = Counter(e.split('/', 1)[1].split('-', 1)[0] for e in ids)
for s, n in sorted(suite.items()):
    print(f'{s}: {n}')
print(f'TOTAL: {len(ids)}')
"
```

Compare against the README "At a glance" table (currently: MiniGrid 71, MiniHack 63, Atari 57, Classics 50, Craftax 43, Procgen 16; total 300).

- [ ] **Step 2: Verify per-suite action counts**

```bash
cd /home/roger/Desktop/rl-world-ascii && uv run python -c "
from glyphbench.core import all_glyphbench_env_ids, make_env
from collections import defaultdict
buckets = defaultdict(set)
for eid in all_glyphbench_env_ids():
    if 'dummy' in eid:
        continue
    suite = eid.split('/', 1)[1].split('-', 1)[0]
    try:
        n = make_env(eid).action_spec.n
        buckets[suite].add(n)
    except Exception as e:
        print(f'SKIP {eid}: {e}')
for s, ns in sorted(buckets.items()):
    rng = f'{min(ns)}' if len(ns) == 1 else f'{min(ns)}-{max(ns)}'
    print(f'{s}: {rng}')
"
```

Compare against README values (MiniGrid 7, MiniHack 22, Atari 3-18, Classics 4-10, Craftax 19/45, Procgen 4-6).

- [ ] **Step 3: Verify install commands and extras**

```bash
grep -A 30 '\[project.optional-dependencies\]' /home/roger/Desktop/rl-world-ascii/pyproject.toml | head -40
```

Compare extras names against README: `[eval]`, `[rl]`, `[all]`. Check package name (`name = ...`). Check Python version (`requires-python = ...`).

- [ ] **Step 4: Verify `make_env` and `load_environment` signatures**

```bash
cd /home/roger/Desktop/rl-world-ascii && uv run python -c "
import inspect
from glyphbench.core import make_env
print('make_env:', inspect.signature(make_env))
import glyphbench
print('load_environment:', inspect.signature(glyphbench.load_environment))
"
```

Compare README quickstart and load_environment example against the actual signatures.

- [ ] **Step 5: Verify memory-mode behavior (one trajectory step vs two)**

```bash
cd /home/roger/Desktop/rl-world-ascii && grep -rn "trajectory step\|memory step\|two trajectory\|one trajectory" src/glyphbench/verifiers_integration/ | head
git log --oneline -- src/glyphbench/verifiers_integration/ | head -10
```

The README says memory mode generations are stored as "one trajectory step" (line 156). Verify against current code state and recent commits (`1c24d37` and `ec3499f`).

- [ ] **Step 6: Verify every script path mentioned in the README exists**

```bash
cd /home/roger/Desktop/rl-world-ascii && for p in eval/run_debug.sh eval/run_full.sh scripts/rl/launch_all.sh scripts/demo_all_envs.py scripts/replay_trajectory.py scripts/record_random_gifs.py configs/rl/qwen35-4b-glyphbench/README.md src/glyphbench/rl/README.md eval/README.md CONTRIBUTING.md .env.cluster.template eval/random_baseline.json scripts/upload_assets.py docs/REPLAY.md; do
  if [ -e "$p" ]; then
    echo "OK   $p"
  else
    echo "MISS $p"
  fi
done
```

Anything `MISS` is a stale README reference.

- [ ] **Step 7: Verify the project layout block matches `src/glyphbench/`**

```bash
ls /home/roger/Desktop/rl-world-ascii/src/glyphbench/ && echo "---" && ls /home/roger/Desktop/rl-world-ascii/
```

Compare against the tree under "Project layout" in the README.

- [ ] **Step 8: Scan for anonymization strings**

```bash
cd /home/roger/Desktop/rl-world-ascii && grep -rn "anon-paper-submission\|Anonymous" README.md eval/ scripts/ src/ docs/ 2>/dev/null | head
```

Anything found is flagged as a "Judgment call" (don't auto-rewrite).

- [ ] **Step 9: Look for shipped features not yet mentioned in the README**

```bash
ls /home/roger/Desktop/rl-world-ascii/src/glyphbench/envs/craftax/docs/ 2>/dev/null && echo "---" && cat /home/roger/Desktop/rl-world-ascii/src/glyphbench/envs/craftax/docs/__init__.py 2>/dev/null | head -30
```

The craftax tutorial deliverable from phase γ (10-chapter LLM-first markdown tutorial) is a candidate; flag if it's not surfaced.

- [ ] **Step 10: Compose `tools/readme-assets/audit.md`**

Write `/home/roger/Desktop/rl-world-ascii/tools/readme-assets/audit.md` with three sections:

```markdown
# README freshness audit — 2026-04-30

## Stale facts to fix
- [list: each item is "claim X in section Y of README → actual Z (source: file:line)"]

## Judgment calls (need user input)
- [list: each item is "X (e.g. anon-paper-submission appears in N places); options: A / B"]

## Confirmed accurate
- [list: 1-line per verified claim]
```

Populate from Steps 1-9 results.

- [ ] **Step 11: Show the user the "Judgment calls" section and pause**

Read the file and surface the section to the user. **Do not proceed past this point until the user has acknowledged each Judgment call** (e.g. confirmed they want to keep `anon-paper-submission`, confirmed the public GitHub URL, etc.). Their decisions become inputs to Tasks 11-18.

- [ ] **Step 12: No commit (audit.md is local-only)**

---

## Task 3: Install asciinema + agg

**Files:** none (system tools)

- [ ] **Step 1: Install asciinema**

On Linux (apt):

```bash
sudo apt-get update && sudo apt-get install -y asciinema
asciinema --version
```

On macOS: `brew install asciinema`. Expected: version >= 2.4.

- [ ] **Step 2: Install agg**

`agg` is the canonical "asciinema cast → GIF" converter. It ships as a single Rust binary.

```bash
which cargo && cargo install --git https://github.com/asciinema/agg agg || (
  ARCH=$(uname -m); OS=$(uname -s | tr 'A-Z' 'a-z')
  curl -L "https://github.com/asciinema/agg/releases/latest/download/agg-${ARCH}-unknown-${OS}" -o /tmp/agg
  chmod +x /tmp/agg && sudo mv /tmp/agg /usr/local/bin/agg
)
agg --version
```

Expected: version >= 1.4.

- [ ] **Step 3: Verify both work end-to-end with a 1-second smoke cast**

```bash
asciinema rec --quiet --cols 80 --rows 10 -c "echo hello && sleep 1" /tmp/smoke.cast
agg /tmp/smoke.cast /tmp/smoke.gif
ls -la /tmp/smoke.gif
```

Expected: `smoke.gif` exists and is non-empty.

- [ ] **Step 4: No commit (system deps)**

---

## Task 4: Build the grid-only renderer

**Files:**
- Create: `tools/readme-assets/render_grid_gif.py` (local-only)

- [ ] **Step 1: Write `tools/readme-assets/render_grid_gif.py`**

```python
#!/usr/bin/env python
"""Throwaway: render per-env grid-only GIFs with per-glyph color palette.

Each frame is the [Grid] block of the env observation, rendered via
Pillow with a hand-picked color per common Unicode glyph. A 1-line
header above the grid shows env_id, turn, action, cumulative reward.

Usage:
    uv run python tools/readme-assets/render_grid_gif.py \\
        --env glyphbench/minigrid-empty-5x5-v0
    uv run python tools/readme-assets/render_grid_gif.py        # all envs
    uv run python tools/readme-assets/render_grid_gif.py --suite craftax
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import glyphbench  # noqa: F401  — registers all envs
from glyphbench.core import all_glyphbench_env_ids, make_env

# ---- per-glyph color palette --------------------------------------------
PALETTE: dict[str, tuple[int, int, int]] = {
    # Player / agent
    "→": (60, 220, 80),  # → right
    "↓": (60, 220, 80),  # ↓ down
    "←": (60, 220, 80),  # ← left
    "↑": (60, 220, 80),  # ↑ up
    "@": (60, 220, 80),
    "☺": (60, 220, 80),  # ☺ player
    # Wall
    "█": (90, 90, 90),   # █ full block
    # Floor
    "·": (50, 50, 50),   # · middle dot
    " ": (15, 15, 15),
    # Goal
    "★": (255, 200, 0),  # ★ star
    "*": (255, 200, 0),
    "✶": (255, 200, 0),  # ✶
    # Water
    "≈": (60, 130, 255),  # ≈
    "~": (60, 130, 255),
    # Door
    "▣": (200, 180, 60),  # ▣
    "D": (200, 180, 60),
    # Key
    "\U0001f511": (220, 80, 220),  # 🔑
    "K": (220, 80, 220),
    # Pickup / health
    "✚": (60, 220, 220),  # ✚
    "+": (60, 220, 220),
    # Hazard
    "✖": (240, 70, 70),   # ✖
    "X": (240, 70, 70),
    "!": (240, 70, 70),
    "⚠": (240, 70, 70),   # ⚠
    # Mob / enemy
    "○": (220, 220, 220), # ○
    "●": (220, 220, 220), # ●
    # Stairs / level
    "▲": (100, 200, 200), # ▲
    "▼": (100, 200, 200), # ▼
}
DEFAULT_COLOR = (200, 200, 200)
BG_COLOR = (15, 15, 15)
HEADER_COLOR = (220, 220, 220)
ACCENT_COLOR = (90, 200, 200)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def _extract_grid(obs: str) -> str:
    """Pull the [Grid] block from a GlyphBench observation string."""
    out: list[str] = []
    in_grid = False
    for line in obs.split("\n"):
        if line.strip() == "[Grid]":
            in_grid = True
            continue
        if in_grid and line.startswith("[") and line.endswith("]"):
            break
        if in_grid:
            out.append(line)
    return "\n".join(out).rstrip("\n")


def _render_frame(
    grid: str,
    header: str,
    font: ImageFont.FreeTypeFont,
) -> Image.Image:
    """One PIL frame: header line + colored grid."""
    char_w = font.getbbox("M")[2]
    line_h = font.size + 4
    grid_lines = grid.split("\n") if grid else [""]
    cols = max((len(line) for line in grid_lines), default=1)
    rows = len(grid_lines)
    img_w = max(600, cols * char_w + 40, len(header) * char_w // 2 + 40)
    img_h = line_h * (rows + 2) + 30
    img = Image.new("RGB", (img_w, img_h), BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw.text((20, 10), header, font=font, fill=HEADER_COLOR)
    y0 = line_h * 2 + 10
    for r, line in enumerate(grid_lines):
        for c, ch in enumerate(line):
            color = PALETTE.get(ch, DEFAULT_COLOR)
            draw.text((20 + c * char_w, y0 + r * line_h), ch, font=font, fill=color)
    return img


def _rollout_frames(env_id: str, seed: int) -> list[tuple[str, str]]:
    """Returns [(grid_text, header_line), ...] for one full random rollout."""
    env = make_env(env_id)
    obs, _ = env.reset(seed)
    action_names = env.action_spec.names
    short = env_id.removeprefix("glyphbench/")
    frames: list[tuple[str, str]] = []
    cum = 0.0
    turn = 0
    while True:
        turn += 1
        a_idx = int(env.rng.integers(0, env.action_spec.n))
        a_name = action_names[a_idx]
        next_obs, reward, terminated, truncated, _ = env.step(a_idx)
        cum += float(reward)
        header = f"{short}  ·  turn {turn}  ·  {a_name}  ·  return {cum:+.2f}"
        frames.append((_extract_grid(obs), header))
        obs = next_obs
        if terminated or truncated:
            tag = "done" if terminated else "truncated"
            header = f"{short}  ·  turn {turn} ({tag})  ·  return {cum:+.2f}"
            frames.append((_extract_grid(obs), header))
            break
    env.close()
    return frames


def render_env_gif(
    env_id: str,
    out_path: Path,
    *,
    font_size: int = 16,
    seed: int = 42,
    duration_ms: int = 200,
) -> int:
    """Render one env. Returns the number of frames written."""
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except OSError:
        font = ImageFont.load_default()
    frames = _rollout_frames(env_id, seed=seed)
    if not frames:
        return 0
    images = [_render_frame(g, h, font) for g, h in frames]
    images[0].save(
        str(out_path),
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    return len(images)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=Path("tools/readme-assets/out/gifs"))
    ap.add_argument("--env", type=str, default=None)
    ap.add_argument("--suite", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--font-size", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.env:
        env_ids = [args.env]
    else:
        env_ids = [e for e in all_glyphbench_env_ids() if "dummy" not in e]
        if args.suite:
            env_ids = [e for e in env_ids if args.suite in e]

    print(f"Rendering {len(env_ids)} env(s) -> {args.output}")
    fails: list[tuple[str, str]] = []
    for i, eid in enumerate(env_ids, 1):
        slug = eid.replace("/", "__")
        out = args.output / f"{slug}.gif"
        if out.exists() and not args.overwrite:
            print(f"[{i}/{len(env_ids)}] {eid}: skip (exists)")
            continue
        try:
            n = render_env_gif(eid, out, font_size=args.font_size, seed=args.seed)
            print(f"[{i}/{len(env_ids)}] {eid}: {n} frames -> {out.name}")
        except Exception as e:
            fails.append((eid, str(e)[:200]))
            print(f"[{i}/{len(env_ids)}] {eid}: ERROR {type(e).__name__}: {e}")
    print(f"\nDone. {len(env_ids) - len(fails)} succeeded, {len(fails)} failed.")
    for eid, err in fails[:10]:
        print(f"  {eid}: {err}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test on a single env**

```bash
cd /home/roger/Desktop/rl-world-ascii && uv run python tools/readme-assets/render_grid_gif.py --env glyphbench/minigrid-empty-5x5-v0 --overwrite
ls -la tools/readme-assets/out/gifs/
```

Expected: `glyphbench__minigrid-empty-5x5-v0.gif` exists, > 50 KB. Open it (`xdg-open` / `open`) and visually confirm: dark background, header line at top, player arrow `→` is green, walls `█` are gray, goal `★` is gold.

- [ ] **Step 3: No commit (file is local-only)**

---

## Task 5: Validation gate (3 sample envs)

**Files:** none (uses Task 4 output dir)

- [ ] **Step 1: Render the three diagnostic envs**

```bash
cd /home/roger/Desktop/rl-world-ascii && uv run python tools/readme-assets/render_grid_gif.py --env glyphbench/craftax-classic-v0 --overwrite
uv run python tools/readme-assets/render_grid_gif.py --env glyphbench/minigrid-empty-5x5-v0 --overwrite
uv run python tools/readme-assets/render_grid_gif.py --env glyphbench/minihack-room-monster-15x15-v0 --overwrite
```

Expected: three GIFs in `tools/readme-assets/out/gifs/`, each rendered without errors.

- [ ] **Step 2: Show the user**

Surface the three GIF paths to the user. They should open them and confirm:
- Player glyphs are colored (green)
- Walls are visible (gray)
- Goals / hazards / items pop with their assigned colors
- Header line is readable
- Frame timing feels OK (200ms per frame)

**Do not proceed to Task 6 until the user explicitly approves.**

If the user requests palette tuning, edit `PALETTE` in `tools/readme-assets/render_grid_gif.py`, re-run Step 1, and re-show.

- [ ] **Step 3: No commit**

---

## Task 6: Render all 300 grid-only GIFs

**Files:** `tools/readme-assets/out/gifs/glyphbench__*.gif` (local-only)

- [ ] **Step 1: Run the full sweep**

```bash
cd /home/roger/Desktop/rl-world-ascii && uv run python tools/readme-assets/render_grid_gif.py --overwrite 2>&1 | tee tools/readme-assets/render_log.txt
```

Expected: ~300 lines of `[N/300] glyphbench/...: K frames -> ....gif`. Total time ~5-10 min. End-of-run summary should report `300 succeeded, 0 failed` (some envs may legitimately fail if their `make_env` raises; record those for re-render with a different seed).

- [ ] **Step 2: Spot-check one env per suite**

```bash
ls tools/readme-assets/out/gifs/ | wc -l
for SUITE in atari classics craftax minigrid minihack procgen; do
  ls tools/readme-assets/out/gifs/glyphbench__${SUITE}-*.gif | head -1
done
```

Open each spot-check GIF and confirm visual quality (colors correct, frame count > 1, file size reasonable). Re-render any failures with `--seed 7` (or another seed) if a particular env produced a degenerate 1-frame GIF.

- [ ] **Step 3: No commit**

---

## Task 7: Build the hero composite GIF

**Files:**
- Create: `tools/readme-assets/build_hero.py` (local-only)
- Create: `tools/readme-assets/out/readme/hero.gif` (local-only)

- [ ] **Step 1: Write `tools/readme-assets/build_hero.py`**

```python
#!/usr/bin/env python
"""Throwaway: stitch 12 chosen env GIFs into a single hero sizzle reel.

Each cell shows ~1.2s of rollout from one env, captioned with its suite
name. Half-second crossfade between cells. Output is a single GIF
sized to the widest source frame (typically ~600-900px), looping
forever.

Pure Python (Pillow + the GIFs we already rendered). No ffmpeg deps.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

GIFS_DIR = Path("tools/readme-assets/out/gifs")
OUT_PATH = Path("tools/readme-assets/out/readme/hero.gif")
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# (env_id, caption shown at the bottom of the cell)
CELLS: list[tuple[str, str]] = [
    ("minigrid-doorkey-6x6-v0",         "MiniGrid · DoorKey-6x6"),
    ("minigrid-multiroom-n4-s5-v0",     "MiniGrid · MultiRoom-N4"),
    ("minihack-room-monster-15x15-v0",  "MiniHack · Room-Monster"),
    ("minihack-quest-easy-v0",          "MiniHack · Quest-Easy"),
    ("atari-pong-v0",                   "Atari · Pong"),
    ("atari-breakout-v0",               "Atari · Breakout"),
    ("classics-snake-medium-v0",        "Classics · Snake"),
    ("classics-sokoban-easy-v0",        "Classics · Sokoban"),
    ("craftax-classic-v0",              "Craftax · Classic"),
    ("craftax-fight-cow-v0",            "Craftax · FightCow"),
    ("procgen-coinrun-v0",              "Procgen · CoinRun"),
    ("procgen-maze-v0",                 "Procgen · Maze"),
]

PER_CELL_FRAMES = 6      # ~1.2s at 200ms/frame
CROSSFADE_FRAMES = 3     # 0.6s crossfade between cells
FRAME_MS = 200
BG = (15, 15, 15)
CAPTION_FILL = (220, 220, 220)
CAPTION_BG = (25, 25, 25)


def _load_gif(path: Path) -> list[Image.Image]:
    """Load all frames of a GIF as RGB images."""
    img = Image.open(path)
    frames: list[Image.Image] = []
    try:
        while True:
            frames.append(img.copy().convert("RGB"))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames


def _resize_to(frames: list[Image.Image], w: int, h: int) -> list[Image.Image]:
    """Letterbox each frame onto a fixed-size canvas (centered)."""
    out: list[Image.Image] = []
    for f in frames:
        canvas = Image.new("RGB", (w, h), BG)
        scale = min(w / f.width, h / f.height)
        new_w = max(1, int(f.width * scale))
        new_h = max(1, int(f.height * scale))
        resized = f.resize((new_w, new_h), Image.LANCZOS)
        canvas.paste(resized, ((w - new_w) // 2, (h - new_h) // 2))
        out.append(canvas)
    return out


def _add_caption(frames: list[Image.Image], caption: str,
                 font: ImageFont.FreeTypeFont) -> list[Image.Image]:
    """Draw a caption strip at the bottom of every frame."""
    out: list[Image.Image] = []
    strip_h = font.size + 12
    for f in frames:
        img = f.copy()
        d = ImageDraw.Draw(img)
        d.rectangle([(0, img.height - strip_h), (img.width, img.height)],
                    fill=CAPTION_BG)
        text_w = font.getlength(caption)
        d.text(((img.width - text_w) / 2, img.height - strip_h + 4),
               caption, font=font, fill=CAPTION_FILL)
        out.append(img)
    return out


def _take_evenly(frames: list[Image.Image], n: int) -> list[Image.Image]:
    """Subsample frames evenly to length n (or repeat the last if too short)."""
    if not frames:
        return []
    if len(frames) >= n:
        idx = [int(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
        return [frames[i] for i in idx]
    return frames + [frames[-1]] * (n - len(frames))


def _crossfade(a: Image.Image, b: Image.Image, alpha: float) -> Image.Image:
    return Image.blend(a, b, alpha)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype(FONT_PATH, 16)
    except OSError:
        font = ImageFont.load_default()

    cell_frames: list[list[Image.Image]] = []
    for env_short, caption in CELLS:
        gif_path = GIFS_DIR / f"glyphbench__{env_short}.gif"
        if not gif_path.exists():
            raise SystemExit(f"missing: {gif_path}  (run Task 6 first)")
        raw = _load_gif(gif_path)
        cell_frames.append((raw, caption))  # type: ignore[arg-type]

    # Pick a uniform cell size: max width × max height across all cells.
    max_w = max(f.width for raw, _ in cell_frames for f in raw)
    max_h = max(f.height for raw, _ in cell_frames for f in raw)
    # Cap to a sensible hero width. Letterbox the rest.
    target_w = min(max_w, 900)
    target_h = min(max_h, 600)

    captioned_cells: list[list[Image.Image]] = []
    for raw, caption in cell_frames:
        resized = _resize_to(raw, target_w, target_h)
        sampled = _take_evenly(resized, PER_CELL_FRAMES)
        captioned = _add_caption(sampled, caption, font)
        captioned_cells.append(captioned)

    out_frames: list[Image.Image] = []
    for i, cell in enumerate(captioned_cells):
        out_frames.extend(cell)
        if i < len(captioned_cells) - 1:
            nxt = captioned_cells[i + 1][0]
            for k in range(1, CROSSFADE_FRAMES + 1):
                alpha = k / (CROSSFADE_FRAMES + 1)
                out_frames.append(_crossfade(cell[-1], nxt, alpha))

    out_frames[0].save(
        str(OUT_PATH),
        save_all=True,
        append_images=out_frames[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
    )
    total_s = len(out_frames) * FRAME_MS / 1000
    print(f"hero.gif: {len(out_frames)} frames @ {FRAME_MS}ms = {total_s:.1f}s")
    print(f"size: {target_w}x{target_h}  ->  {OUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
cd /home/roger/Desktop/rl-world-ascii && uv run python tools/readme-assets/build_hero.py
ls -la tools/readme-assets/out/readme/hero.gif
```

Expected: a single `hero.gif`, dimensions printed (e.g. `900x600`), total ~12-15s at 200ms/frame.

- [ ] **Step 3: Inspect**

Open `hero.gif` and confirm: 12 envs visible in sequence, captions readable at bottom of each cell, smooth crossfades, no janky letterboxing or color shifts. If a specific cell looks weak, swap its env in `CELLS` for a different pick from that suite, re-run.

- [ ] **Step 4: No commit**

---

## Task 8: Record `demo_all_envs.gif`

**Files:**
- Create: `tools/readme-assets/out/readme/demo_all_envs.cast` (local-only)
- Create: `tools/readme-assets/out/readme/demo_all_envs.gif` (local-only)

- [ ] **Step 1: Record the cast**

```bash
cd /home/roger/Desktop/rl-world-ascii && asciinema rec --quiet --cols 140 --rows 40 \
  --idle-time-limit 0.5 \
  --command "uv run python scripts/demo_all_envs.py --env glyphbench/craftax-classic-v0 --delay 0.1" \
  tools/readme-assets/out/readme/demo_all_envs.cast
```

The recording auto-terminates when the env terminates or truncates. Expected duration: 20-60s (craftax-classic random rollout under default `max_turns`). Idle time capped at 0.5s.

- [ ] **Step 2: Convert to GIF**

```bash
agg --theme monokai --font-size 14 --speed 1.5 \
  tools/readme-assets/out/readme/demo_all_envs.cast \
  tools/readme-assets/out/readme/demo_all_envs.gif
ls -la tools/readme-assets/out/readme/demo_all_envs.gif
```

Expected: GIF exists, < 5 MB, looks crisp. `--speed 1.5` gently accelerates so the README doesn't feel sluggish.

- [ ] **Step 3: Inspect**

Open it and confirm: full multi-panel TUI is visible (header / system prompt panel / grid panel / recent actions / step / HUD / legend / action / env feedback), updates smoothly across the rollout, glyphs render correctly. If the font looks wrong, retry with `--font-family "DejaVu Sans Mono"`.

- [ ] **Step 4: No commit**

---

## Task 9: Record `gb_replay.gif`

**Files:**
- Create: `tools/readme-assets/out/readme/gb_replay.cast` (local-only)
- Create: `tools/readme-assets/out/readme/gb_replay.gif` (local-only)

**Prerequisite:** A saved trajectory for `glyphbench/minigrid-empty-5x5-v0` exists at `cluster_manager/results/evals/glyphbench--Qwen--Qwen3.5-4B/0a6cef3f/results.jsonl`. Verify before proceeding:

```bash
grep -l "minigrid-empty-5x5-v0" /home/roger/Desktop/rl-world-ascii/cluster_manager/results/evals/*/*/results.jsonl
```

Expected: at least one path. If none, fall back to `glyphbench/atari-alien-v0` (the other available trajectory) and update the README references later.

- [ ] **Step 1: Dry-run `gb replay` non-recorded first to confirm the panels render**

```bash
cd /home/roger/Desktop/rl-world-ascii && uv run glyphbench replay cluster_manager/results --env glyphbench/minigrid-empty-5x5-v0 --pause
```

Press `→` a couple of times to step, `q` to exit. Confirm the TUI looks right (no stale panels, no parse-error red borders unexpectedly). Resize your terminal to ~140 cols × 40 rows before recording for consistency.

- [ ] **Step 2: Record the cast (interactive — user presses keys)**

```bash
cd /home/roger/Desktop/rl-world-ascii && asciinema rec --cols 140 --rows 40 \
  tools/readme-assets/out/readme/gb_replay.cast
```

Then in the recorded session, run:

```bash
uv run glyphbench replay cluster_manager/results --env glyphbench/minigrid-empty-5x5-v0 --pause
```

**Hotkey sequence (press deliberately, ~1 sec between each):**

```
→  →  →                  (step three turns)
s                        (open system prompt in pager)
G                        (jump to bottom of pager)
q                        (exit pager)
→                        (step)
r                        (open reasoning)
q                        (exit pager)
m                        (open memory diff)
q                        (exit pager)
→  →                     (step two more)
q                        (exit replay)
```

After replay exits, type `exit` to end the asciinema recording (or press `Ctrl-D`).

- [ ] **Step 3: Convert to GIF**

```bash
agg --theme monokai --font-size 14 --speed 1.0 \
  tools/readme-assets/out/readme/gb_replay.cast \
  tools/readme-assets/out/readme/gb_replay.gif
ls -la tools/readme-assets/out/readme/gb_replay.gif
```

Expected: GIF exists, < 5 MB.

- [ ] **Step 4: Inspect**

Open it. Required elements visible: header bar, system prompt panel, grid panel, side panels (HUD/legend/action), pager popping out for system/reasoning/memory and being dismissed back to the TUI. If the recording is too long, edit `gb_replay.cast` to trim leading/trailing dead time (`asciinema cat` shows the timing — manually adjust if needed) or re-record more efficiently.

- [ ] **Step 5: No commit**

---

## Task 10: Upload all assets to HF

**Files:** none (uses existing `scripts/upload_assets.py`)

- [ ] **Step 1: Upload the 300 grid-only GIFs**

```bash
cd /home/roger/Desktop/rl-world-ascii && uv run python scripts/upload_assets.py \
  --src tools/readme-assets/out/gifs \
  --dst gifs \
  --commit-message "Regenerate all 300 random-agent GIFs with colored Unicode glyph palette"
```

Expected: hf upload progress, final "Uploaded 300 files" message. If any file fails (rate limit), re-run — the upload skips files already present on HF.

- [ ] **Step 2: Upload the 3 README assets**

```bash
cd /home/roger/Desktop/rl-world-ascii && uv run python scripts/upload_assets.py \
  --src tools/readme-assets/out/readme \
  --dst readme \
  --commit-message "Add README hero, demo, and gb-replay walkthrough GIFs"
```

Expected: 3 files uploaded.

- [ ] **Step 3: Verify all README-referenced URLs resolve**

```bash
HF_BASE="https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main"
for asset in readme/hero.gif readme/demo_all_envs.gif readme/gb_replay.gif \
             gifs/glyphbench__minigrid-doorkey-6x6-v0.gif \
             gifs/glyphbench__atari-pong-v0.gif \
             gifs/glyphbench__craftax-classic-v0.gif; do
  status=$(curl -sI "${HF_BASE}/${asset}" -o /dev/null -w "%{http_code}")
  echo "${status} ${asset}"
done
```

Expected: every line starts with `200 ` or `302 `.

- [ ] **Step 4: No commit (HF state changes only)**

---

## Task 11: Write `docs/OBSERVATION_FORMAT.md`

**Files:**
- Create: `docs/OBSERVATION_FORMAT.md`

- [ ] **Step 1: Read the audit findings relevant to observation format and harness**

Re-read `tools/readme-assets/audit.md` for any stale claims about the observation format, harness behavior, frame stacking, or memory mode. These drive the content.

- [ ] **Step 2: Read the current source as ground truth**

```bash
ls /home/roger/Desktop/rl-world-ascii/src/glyphbench/core/
```

Read `core/observation.py` (or equivalent) and `verifiers_integration/multi_turn.py` to ground claims about what the model sees per turn.

- [ ] **Step 3: Write `docs/OBSERVATION_FORMAT.md`**

Required sections (each ~5-15 lines, scaled to actual content):

```markdown
# Observation format and harness

## What the agent sees per turn

The harness shows the model a single text string composed of these blocks:

- `[Legend]` — glyph → meaning mapping (deduped across the rollout).
- `[Grid]` — the 2D Unicode grid (the only required channel).
- `[Message]` — optional per-turn narrative event.
- `[Actions]` — the action vocabulary the model picks from this turn.

Envs may also compute a `[HUD]` (HP, inventory, score, etc.) for their `info`
dict and trajectory logs. The harness deliberately does NOT show the HUD to
the model — privileged state must be encoded in the visible grid for the
agent to reason about.

## Single-codepoint Unicode glyphs

Every cell is exactly one codepoint. No symbol collides within a suite. Common
glyphs: `█` walls, `→ ↓ ← ↑` player direction, `★` goal, `≈` water, `▣` door.

## System prompt

`env.system_prompt()` returns a compact rules / actions / reward / termination
description, ready to pass as a system message to any LLM.

## Frame-stacked history (n_frames)

`load_environment(..., n_frames=N)` joins the last N observation strings into
a single user message per turn. Default: `n_frames=0` (stateless / pure
Markov).

## Memory mode

`load_environment(..., use_memory=True)` adds an opt-in memory scaffold:
each step uses TWO model generations — one for the action, one for a concise
memory update conditioned on action / reward / done / next-observation.

[Document current behavior of the memory-step split per audit Task 2 Step 5;
either "stored as one trajectory step" or "stored as two trajectory steps".
Resolve from the audit before writing.]

`memory_update_max_tokens` overrides only the second generation's token
limit; defaults to the action sampling limit. Memory-aware trajectories
show previous and updated memory in `glyphbench replay`.

## Determinism

All observations are deterministic — identical seeds produce identical
trajectories. Privileged state lives in the grid; the model has no
side-channel.
```

Fill in all bracketed `[...]` placeholders before writing.

- [ ] **Step 4: Verify links and references**

```bash
cd /home/roger/Desktop/rl-world-ascii && grep -E "(load_environment|make_env|n_frames|use_memory)" docs/OBSERVATION_FORMAT.md
```

Each occurrence should match the actual API (verified in Task 2 Step 4).

- [ ] **Step 5: Commit**

```bash
cd /home/roger/Desktop/rl-world-ascii && git add docs/OBSERVATION_FORMAT.md
git commit -m "$(cat <<'EOF'
docs: add OBSERVATION_FORMAT.md (observation channels + harness + memory mode)

Single source of truth for what the agent sees per turn, the system
prompt API, frame-stacking, memory mode, and determinism guarantees.
Pulled out of the main README to keep the README compact and route
depth here.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Write `docs/INTEGRATION.md`

**Files:**
- Create: `docs/INTEGRATION.md`

- [ ] **Step 1: Write the file**

Required sections:

```markdown
# Use GlyphBench with your own agent

## Direct game loop (Python)

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
```

## What `your_agent` receives

- `obs` is a single text string (Legend + Grid + Message + Actions).
- `env.action_spec.names` is the list of action names valid this turn.
- Return either an integer action index or the action name string —
  both are accepted by `env.step`.

## Loading as a verifiers env (for vf-eval / RL)

```python
import glyphbench
vf_env = glyphbench.load_environment(
    task_id="glyphbench/minigrid-empty-5x5-v0",
    num_episodes=5,
    n_frames=0,
    max_output_tokens=8192,
    use_memory=False,
)
```

Returns a `verifiers.MultiTurnEnv` ready for `vf.evaluate(...)` or RL training.

## Inspecting trajectories

Save your agent's trajectory as a JSONL file (one step per line, see
`scripts/replay_trajectory.py` for the schema), then replay it with the rich
TUI:

```bash
uv run glyphbench replay path/to/runs_dir --env glyphbench/<env_id>
```

Or render it as a GIF:

```bash
uv run python scripts/replay_trajectory.py trajectory.jsonl --gif out.gif
```

See `docs/REPLAY.md` for the full hotkey reference and panel layout.
```

- [ ] **Step 2: Commit**

```bash
cd /home/roger/Desktop/rl-world-ascii && git add docs/INTEGRATION.md
git commit -m "$(cat <<'EOF'
docs: add INTEGRATION.md (use GlyphBench with your own agent)

Cookbook for direct-loop and verifiers integration paths, plus
trajectory inspection. Pulled out of the main README to route depth
here.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Write `docs/ARCHITECTURE.md`

**Files:**
- Create: `docs/ARCHITECTURE.md`

- [ ] **Step 1: Confirm the directory tree**

```bash
ls /home/roger/Desktop/rl-world-ascii/src/glyphbench/
ls /home/roger/Desktop/rl-world-ascii/
```

- [ ] **Step 2: Write the file**

```markdown
# Architecture

## Top-level layout

```
src/glyphbench/
    core/                  # BaseGlyphEnv, GridObservation, ActionSpec, registry
    envs/                  # 6 suites, 300 envs (atari, classics, craftax,
                           #   minigrid, minihack, procgen)
    verifiers_integration/ # prompt builder, parser, multi-turn env, rubric
    plotting/              # parquet loaders + paper-figure generators
    rl/                    # custom advantage / loss hooks for prime-rl
    cli.py                 # `glyphbench replay` TUI

eval/                      # vf-eval wrappers, random-agent baseline
configs/                   # endpoint registry, prime-rl training configs
cluster_manager/           # SLURM multi-cluster experiment manager
scripts/                   # demo, trajectory replay, GIF export, upload
docs/                      # this directory
docs/leaderboard/          # GitHub Pages site (leaderboard + rollout gallery)
```

## Key boundaries

- Envs declare `system_prompt()`, `reset()`, `step()`. They never know
  about the harness, the LLM, or verifiers — they're pure simulators with
  text observations.
- The harness (`verifiers_integration/`) composes the system prompt, parses
  the model's action XML, runs `env.step`, and packages everything for
  the verifiers `MultiTurnEnv` API.
- `cli.py` (the `gb replay` TUI) reads `verifiers` `results.jsonl` files and
  renders them with Rich. It does not depend on the envs at runtime
  (cached `_spec_for_env_id` lazily instantiates one per env_id).
- `rl/` plugs into prime-rl as a custom advantage + loss; per-env Welford
  tracks reward statistics for normalization. See `src/glyphbench/rl/README.md`.
```

- [ ] **Step 3: Verify the tree against actual filesystem**

```bash
cd /home/roger/Desktop/rl-world-ascii && for d in src/glyphbench/core src/glyphbench/envs src/glyphbench/verifiers_integration src/glyphbench/plotting src/glyphbench/rl src/glyphbench/cli.py eval configs cluster_manager scripts docs docs/leaderboard; do
  if [ -e "$d" ]; then echo "OK   $d"; else echo "MISS $d"; fi
done
```

Expected: every line `OK`. Fix any `MISS` by adjusting the doc.

- [ ] **Step 4: Commit**

```bash
cd /home/roger/Desktop/rl-world-ascii && git add docs/ARCHITECTURE.md
git commit -m "$(cat <<'EOF'
docs: add ARCHITECTURE.md (project layout + key boundaries)

One-page directory tree + 1-2 line per area. Pulled out of the main
README to keep the README compact.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Write `scripts/README.md`

**Files:**
- Create: `scripts/README.md`

- [ ] **Step 1: List the scripts**

```bash
ls /home/roger/Desktop/rl-world-ascii/scripts/*.py /home/roger/Desktop/rl-world-ascii/scripts/*.sh 2>/dev/null
```

- [ ] **Step 2: Write the file**

For each script in `scripts/`, document one usage line. Pull the canonical command from each script's docstring header.

```markdown
# scripts/

Top-level operator scripts. Each one's docstring has the full flag list;
this index just shows the canonical invocation.

## Demo + replay

- **`demo_all_envs.py`** — Watch a uniform-random agent play any env in the
  same TUI layout `gb replay` uses.
  ```bash
  uv run python scripts/demo_all_envs.py --suite minigrid --delay 0.1
  uv run python scripts/demo_all_envs.py --env glyphbench/craftax-classic-v0 --pause
  ```

- **`replay_trajectory.py`** — Replay a saved trajectory JSONL with color, or
  export it as a GIF.
  ```bash
  uv run python scripts/replay_trajectory.py path/to/trajectory.jsonl
  uv run python scripts/replay_trajectory.py path/to/trajectory.jsonl --gif out.gif
  ```

## Asset generation

- **`record_random_gifs.py`** — Render a random-agent GIF for every env using
  the `replay_trajectory.export_gif` legacy renderer (kept for backward
  compatibility; the public README assets are now produced by a separate
  pipeline).
  ```bash
  uv run python scripts/record_random_gifs.py --output docs/leaderboard/gifs/
  ```

- **`upload_assets.py`** — Upload a directory of files to the GlyphBench HF
  dataset repo.
  ```bash
  uv run python scripts/upload_assets.py --src docs/leaderboard/gifs --dst gifs
  ```

## Leaderboard / catalog

- **`build_leaderboard.py`** — Aggregate `cluster_manager/results/` into the
  GitHub Pages site under `docs/leaderboard/`.
- **`generate_env_catalog.py`** — Refresh `docs/ENVIRONMENTS.md` from the
  current registry.

## Misc

- **`play_curses.py`** / **`play_interactive.py`** / **`play_random.py`** —
  Manual play / random rollout in plain stdout (legacy; use `demo_all_envs.py`
  for the rich panel layout).
- **`build_sif.sh`** — Build the apptainer image used on SLURM clusters.
- **`terminal_colors.py`** — Quick color-table reference for ANSI 256 / truecolor.
- **`rl/`** — RL-pipeline launchers; see `scripts/rl/README.md`.
```

- [ ] **Step 3: Verify each mentioned script exists**

```bash
cd /home/roger/Desktop/rl-world-ascii && for s in scripts/demo_all_envs.py scripts/replay_trajectory.py scripts/record_random_gifs.py scripts/upload_assets.py scripts/build_leaderboard.py scripts/generate_env_catalog.py scripts/play_curses.py scripts/play_interactive.py scripts/play_random.py scripts/build_sif.sh scripts/terminal_colors.py scripts/rl/; do
  [ -e "$s" ] && echo "OK   $s" || echo "MISS $s"
done
```

Adjust the doc for any `MISS`.

- [ ] **Step 4: Commit**

```bash
cd /home/roger/Desktop/rl-world-ascii && git add scripts/README.md
git commit -m "$(cat <<'EOF'
docs: add scripts/README.md (index of top-level operator scripts)

One-page reference for all scripts/*.py with canonical invocations.
Pulled out of the main README.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Refresh `docs/REPLAY.md` and embed `gb_replay.gif`

**Files:**
- Modify: `docs/REPLAY.md`

- [ ] **Step 1: Read the current file**

```bash
head -100 /home/roger/Desktop/rl-world-ascii/docs/REPLAY.md
```

- [ ] **Step 2: Verify currency against `src/glyphbench/cli.py`**

```bash
cd /home/roger/Desktop/rl-world-ascii && grep -nE "def _read_one_key|--pause|--delay|--list|--episode" src/glyphbench/cli.py | head -20
```

Confirm the README's flag list and hotkey list match the current CLI. Fix any drift.

- [ ] **Step 3: Insert the GIF at the top, just under the title**

After the file's `# `gb replay`` heading, add:

```markdown
![gb replay walkthrough](https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/readme/gb_replay.gif)

*Pause-mode hotkeys: `s` system prompt · `r` reasoning · `m` memory diff · `←/→` step · `q` next.*
```

- [ ] **Step 4: Commit**

```bash
cd /home/roger/Desktop/rl-world-ascii && git add docs/REPLAY.md
git commit -m "$(cat <<'EOF'
docs(REPLAY): embed gb_replay.gif walkthrough at top + refresh against current CLI

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Refresh `eval/README.md`

**Files:**
- Modify (or create): `eval/README.md`

- [ ] **Step 1: Read the current file (if it exists)**

```bash
test -f /home/roger/Desktop/rl-world-ascii/eval/README.md && cat /home/roger/Desktop/rl-world-ascii/eval/README.md || echo "(does not exist)"
ls /home/roger/Desktop/rl-world-ascii/eval/
```

- [ ] **Step 2: Verify the contents of `eval/run_debug.sh` and `eval/run_full.sh`**

```bash
cat /home/roger/Desktop/rl-world-ascii/eval/run_debug.sh
echo "---"
cat /home/roger/Desktop/rl-world-ascii/eval/run_full.sh
```

- [ ] **Step 3: Write or rewrite `eval/README.md`**

Required sections (absorbing the "Running LLM evaluations" + "Scoring" + parts of the random-baseline content from the main README):

```markdown
# eval/

Verifiers-driven evaluation of every GlyphBench env against any
OpenAI-compatible endpoint.

## Quick start

```bash
# Wire-check: 1 env, 1 episode
bash eval/run_debug.sh

# Full sweep: all 300 envs, configurable via $EPISODES / $MODEL
bash eval/run_full.sh
```

Both scripts assume an OpenAI-compatible server is reachable at
`http://localhost:8000/v1`. Easiest local server:

```bash
uv run vllm serve Qwen/Qwen3.5-4B --port 8000
```

## CLI flags

[Pull from run_debug.sh / run_full.sh — document MODEL, EPISODES,
N_FRAMES, MAX_TOKENS, USE_MEMORY env vars and any positional args.]

## Random-agent baseline

A reproducible zero-skill reference is shipped at
`eval/random_baseline.json`. Regenerate via:

```bash
uv run python eval/random_baseline.py
```

## Scoring

GlyphBench reports **raw episodic return per (env, model)**. There is no
benchmark-wide normalised score: per-task per-model means are published
raw and downstream analyses choose their own aggregation.
```

Fill in `[Pull from run_debug.sh ...]` from Step 2 inspection.

- [ ] **Step 4: Commit**

```bash
cd /home/roger/Desktop/rl-world-ascii && git add eval/README.md
git commit -m "$(cat <<'EOF'
docs(eval): refresh eval/README.md against current run scripts + absorb scoring section

Pulls the running-LLM-evals + scoring + random-baseline detail out of the
main README and centralizes it here.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: Refresh `src/glyphbench/rl/README.md`

**Files:**
- Modify (or create): `src/glyphbench/rl/README.md`

- [ ] **Step 1: Read the current file and sibling code**

```bash
test -f /home/roger/Desktop/rl-world-ascii/src/glyphbench/rl/README.md && cat /home/roger/Desktop/rl-world-ascii/src/glyphbench/rl/README.md || echo "(does not exist)"
ls /home/roger/Desktop/rl-world-ascii/src/glyphbench/rl/
ls /home/roger/Desktop/rl-world-ascii/scripts/rl/
ls /home/roger/Desktop/rl-world-ascii/configs/rl/
```

- [ ] **Step 2: Verify recent state (post RL pipeline merge)**

```bash
cd /home/roger/Desktop/rl-world-ascii && git log --oneline -- src/glyphbench/rl/ scripts/rl/ configs/rl/ | head -20
```

- [ ] **Step 3: Write or rewrite the file**

Required sections (capturing the design notes for custom advantage + loss + per-env Welford + memory-step split, all post-merge as of commits `452c5fd`, `1c24d37`, `ec3499f`):

```markdown
# RL fine-tuning hooks for prime-rl

GlyphBench ships a custom advantage estimator and loss for prime-rl
that:

- Tracks **per-env reward statistics** with online Welford updates
  (env-level normalisation, not batch-level).
- Implements a **custom GRPO loss** over the (env, model) sample
  groups so per-env reward variance is decoupled.
- Splits **memory-mode generations into two trajectory steps** so
  prime-rl's pretokenizer accepts them and action-tokens train
  alongside memory-update tokens with the same task reward.

## Files

- `advantage.py` — per-env Welford + advantage computation.
- `loss.py` — custom GRPO loss head.
- `pipeline.py` — wiring into prime-rl's training loop.
[Adjust file list to actual contents of src/glyphbench/rl/.]

## Configs

`configs/rl/qwen35-4b-glyphbench/` carries the canonical training config
(thinking on, 8K action + 4K memory token budgets, memory mode on).
See `configs/rl/qwen35-4b-glyphbench/README.md`.

## Launch

From the trainer node:

```bash
bash scripts/rl/launch_all.sh
```

This brings up the inference, environment, and trainer components
described in `scripts/rl/README.md` (operator guide).
```

Fill in `[Adjust file list ...]` from Step 1.

- [ ] **Step 4: Commit**

```bash
cd /home/roger/Desktop/rl-world-ascii && git add src/glyphbench/rl/README.md
git commit -m "$(cat <<'EOF'
docs(rl): refresh src/glyphbench/rl/README.md against post-merge state

Documents per-env Welford, custom GRPO loss, and memory-step split as
landed in 452c5fd / 1c24d37 / ec3499f. Pulls RL pipeline detail out of
the main README.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: Rewrite the main `README.md`

**Files:**
- Modify: `README.md` (full rewrite)

- [ ] **Step 1: Re-read the audit's "Stale facts to fix"**

```bash
cat /home/roger/Desktop/rl-world-ascii/tools/readme-assets/audit.md | head -60
```

Every item in "Stale facts to fix" must be addressed in the rewrite. Every "Judgment call" must reflect the user's resolution from Task 2 Step 11.

- [ ] **Step 2: Write the new README.md**

Replace the entire file contents with this skeleton, expanded with audit-driven concrete values:

```markdown
<div align="center">

# GlyphBench

**A benchmark of [N] text-rendered reinforcement-learning environments for evaluating LLM agents on sequential decision-making.**

![hero](https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/readme/hero.gif)

[Leaderboard](https://roger-creus.github.io/glyphbench/leaderboard/) · [Paper (coming soon)](#) · [Quickstart](#quickstart) · [Contributing](CONTRIBUTING.md)

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![vLLM compatible](https://img.shields.io/badge/inference-vLLM-orange)
![Verifiers](https://img.shields.io/badge/eval-verifiers-purple)
[![Leaderboard](https://img.shields.io/badge/leaderboard-live-green)](https://roger-creus.github.io/glyphbench/leaderboard/)

</div>

Every environment renders its state as a Unicode text grid with a legend and discrete named actions. The agent sees only the grid — no privileged state-channel — so every game-relevant fact must be readable off the glyphs themselves. Observations are deterministic (seeded), making results fully reproducible.

## At a glance

| Suite | Envs | What it tests | Actions |
|---|---:|---|---:|
| MiniGrid | [N] | Grid navigation, key/door puzzles, dynamic obstacles, memory | [A] |
| MiniHack | [N] | NetHack-inspired dungeons, combat, items, skills | [A] |
| Atari | [N] | Classic arcade (Pong, Breakout, Space Invaders, …) | [A] |
| Classics | [N] | Snake, Sokoban, Minesweeper, Sudoku, Nim, … | [A] |
| Craftax | [N] | Open-world survival + crafting, dungeon floors, focused sub-tasks | [A] |
| Procgen | [N] | Procedurally generated platformers, shooters, mazes | [A] |

[Replace [N] and [A] with actual values from Task 2 Step 1 / Step 2.]

## Browse the suites

| | | |
|:---:|:---:|:---:|
| ![](https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__minigrid-doorkey-6x6-v0.gif) | ![](https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__minigrid-multiroom-n4-s5-v0.gif) | ![](https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__minihack-room-monster-15x15-v0.gif) |
| **MiniGrid · DoorKey** | **MiniGrid · MultiRoom** | **MiniHack · Room-Monster** |
| ![](https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__minihack-quest-easy-v0.gif) | ![](https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__atari-pong-v0.gif) | ![](https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__atari-breakout-v0.gif) |
| **MiniHack · Quest** | **Atari · Pong** | **Atari · Breakout** |
| ![](https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__classics-snake-medium-v0.gif) | ![](https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__classics-sokoban-easy-v0.gif) | ![](https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__craftax-classic-v0.gif) |
| **Classics · Snake** | **Classics · Sokoban** | **Craftax · Classic** |
| ![](https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__craftax-fight-cow-v0.gif) | ![](https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__procgen-coinrun-v0.gif) | ![](https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/gifs/glyphbench__procgen-maze-v0.gif) |
| **Craftax · FightCow** | **Procgen · CoinRun** | **Procgen · Maze** |

→ [Browse all 300 environments](https://roger-creus.github.io/glyphbench/leaderboard/gallery.html)

## Install

```bash
uv add glyphbench                    # core (environments only)
uv add "glyphbench[eval]"            # + verifiers + vLLM (eval + RL integration)
uv add "glyphbench[all]"             # + providers, analysis, dev tooling
```

## Quickstart

```python
import glyphbench
from glyphbench.core import make_env

env = make_env("glyphbench/minigrid-empty-5x5-v0")
obs, info = env.reset(42)
print(obs)
# Or: load as a verifiers env for eval / RL
vf_env = glyphbench.load_environment(task_id="glyphbench/minigrid-empty-5x5-v0")
```

## Tools

### Trajectory replay

![gb replay](https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/readme/gb_replay.gif)

`glyphbench replay` is a rich TUI for stepping through saved rollouts: per-turn grid + reasoning + memory + HUD, with hotkey hops to a pager for full system prompt / reasoning / memory views. → [docs/REPLAY.md](docs/REPLAY.md)

### Interactive demo

![demo_all_envs](https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/readme/demo_all_envs.gif)

`scripts/demo_all_envs.py` runs a uniform-random agent through every env in the same TUI layout `gb replay` uses. → [scripts/README.md](scripts/README.md)

## Documentation

- [Observation format · harness](docs/OBSERVATION_FORMAT.md)
- [LLM evaluation (vLLM / verifiers)](eval/README.md)
- [RL training (prime-rl)](src/glyphbench/rl/README.md)
- [Trajectory replay tool](docs/REPLAY.md)
- [Interactive demo & scripts](scripts/README.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Use with your own agent](docs/INTEGRATION.md)
- [Contributing](CONTRIBUTING.md)

## Citation

```bibtex
@article{glyphbench2026,
  title   = {GlyphBench: A Unified Benchmark for Evaluating LLM Agents on Sequential Decision-Making},
  author  = {Anonymous},
  year    = {2026},
}
```

## License

MIT
```

Replace each `[N]` and `[A]` placeholder with the verified values from Task 2 Step 1/2. If the user resolved the GitHub URL judgment-call as something other than `roger-creus/glyphbench`, update all `roger-creus.github.io` and asset URLs accordingly. If they resolved the citation anonymization, update the bibtex `author = ` field.

- [ ] **Step 3: Verify line count is in target range**

```bash
wc -l /home/roger/Desktop/rl-world-ascii/README.md
```

Expected: 100-180 lines. If it grew past 180, look for content to push to the subdir docs.

- [ ] **Step 4: Verify all asset URLs resolve**

```bash
cd /home/roger/Desktop/rl-world-ascii && grep -oE 'https://huggingface\.co/[^)]+' README.md | sort -u | while read url; do
  status=$(curl -sI "$url" -o /dev/null -w "%{http_code}")
  echo "$status $url"
done
```

Expected: every URL returns `200` or `302`.

- [ ] **Step 5: Verify all relative links resolve to existing files**

```bash
cd /home/roger/Desktop/rl-world-ascii && grep -oE '\]\([a-zA-Z][^)]*\.md\)|\]\([a-zA-Z][^)]*/[^)]*\)' README.md | sed -E 's/^\]\(([^)]+)\)$/\1/' | while read p; do
  [ -e "$p" ] && echo "OK   $p" || echo "MISS $p"
done
```

Expected: every line `OK`. Anything `MISS` is a link to a doc/file that doesn't exist — fix the link or create the target.

- [ ] **Step 6: Commit**

```bash
cd /home/roger/Desktop/rl-world-ascii && git add README.md
git commit -m "$(cat <<'EOF'
docs(README): rewrite for public release — hero GIF, gallery, tools, doc map

Compact (~150 line) main README with:
  - centered hero block (title, tagline, hero.gif, badges, quick-nav)
  - 4×3 gallery of 12 representative envs with HF-hosted GIFs
  - tools section embedding gb_replay.gif + demo_all_envs.gif
  - documentation map routing to per-area subdir docs

All depth content (observation format, eval, RL, integration, scripts,
architecture) lives in the per-area docs added in earlier commits.
Asset URLs point at the HF dataset anon-paper-submission/glyphbench-assets.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: Public-release readiness check

**Files:** none (verification only)

- [ ] **Step 1: Verify the README renders correctly via gh markdown preview**

If `gh` CLI is configured for this repo:

```bash
cd /home/roger/Desktop/rl-world-ascii && gh markdown-preview README.md 2>/dev/null || (
  echo "gh markdown-preview unavailable; use grip instead:"
  python -m grip README.md --browser 2>/dev/null || echo "install grip with: pip install grip"
)
```

Open the rendered page. Verify:
- Hero block is centered.
- Hero GIF loads.
- Gallery table renders as a 4×3 grid with captions under each cell.
- Both tool GIFs (gb_replay, demo_all_envs) load.
- All section anchors / links jump correctly.

- [ ] **Step 2: Curl-HEAD every asset URL referenced in the README + subdir docs**

```bash
cd /home/roger/Desktop/rl-world-ascii && grep -rohE 'https://huggingface\.co/[^)" ]+' README.md docs/ eval/ scripts/README.md src/glyphbench/rl/README.md 2>/dev/null | sort -u | while read url; do
  status=$(curl -sI "$url" -o /dev/null -w "%{http_code}")
  echo "$status $url"
done
```

Expected: every URL `200` or `302`. Re-upload anything that returns `404`.

- [ ] **Step 3: Verify required public-release files exist**

```bash
cd /home/roger/Desktop/rl-world-ascii && for f in LICENSE CONTRIBUTING.md README.md pyproject.toml; do
  [ -e "$f" ] && echo "OK   $f" || echo "MISS $f"
done
```

If `LICENSE` is `MISS`, create it before going public — MIT text. If `CONTRIBUTING.md` is `MISS`, the README should not link to it (already verified in Task 18 Step 5, but check again).

- [ ] **Step 4: Scan all touched docs for leftover scratch / TODOs / placeholders**

```bash
cd /home/roger/Desktop/rl-world-ascii && grep -nE "TODO|FIXME|TBD|XXX|\[N\]|\[A\]|\[Adjust|\[Replace|\[Pull from|\[Document" README.md docs/OBSERVATION_FORMAT.md docs/INTEGRATION.md docs/ARCHITECTURE.md docs/REPLAY.md scripts/README.md eval/README.md src/glyphbench/rl/README.md 2>/dev/null
```

Expected: no output. Anything found is an unfilled placeholder — fill it before continuing.

- [ ] **Step 5: Push the branch and show the user the GitHub-rendered URL**

```bash
cd /home/roger/Desktop/rl-world-ascii && BRANCH="readme/public-release-shine"
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
git push -u origin "$BRANCH"
echo "Open: https://github.com/$(git config --get remote.origin.url | sed -E 's|.*github.com[:/]([^/]+/[^/.]+)(\.git)?|\1|')/blob/${BRANCH}/README.md"
```

Surface the URL to the user. **Do not proceed until they confirm the rendered README looks correct.**

If they request changes, loop back to Task 18 Step 2 (or the relevant subdir doc task), revise, re-push.

---

## Task 20: Cleanup the throwaway workspace

**Files:**
- Delete: `tools/readme-assets/` (local-only directory)

- [ ] **Step 1: Confirm nothing inside is tracked**

```bash
cd /home/roger/Desktop/rl-world-ascii && git ls-files tools/readme-assets/ | wc -l
```

Expected: `0`.

- [ ] **Step 2: Remove the directory**

```bash
rm -rf /home/roger/Desktop/rl-world-ascii/tools/readme-assets/
ls /home/roger/Desktop/rl-world-ascii/tools/ 2>/dev/null || echo "(no tools/ dir)"
```

If `tools/` was created only for this work and is now empty, remove it too:

```bash
rmdir /home/roger/Desktop/rl-world-ascii/tools/ 2>/dev/null || true
```

- [ ] **Step 3: Remove the local-only ignore line**

```bash
cd /home/roger/Desktop/rl-world-ascii && sed -i '/^tools\/readme-assets\/$/d' .git/info/exclude
cat .git/info/exclude
```

- [ ] **Step 4: Final status check**

```bash
cd /home/roger/Desktop/rl-world-ascii && git status --short
```

Expected: only `?? Craftax-main/` remains (the same untracked dir from before this work). All README + subdir doc changes should already be committed by earlier tasks.

- [ ] **Step 5: Merge the branch (or open PR — user's call)**

If the user wants to merge directly:

```bash
cd /home/roger/Desktop/rl-world-ascii && git checkout main && git merge --no-ff readme/public-release-shine
git push origin main
```

If the user wants a PR (recommended for public-release reviews):

```bash
cd /home/roger/Desktop/rl-world-ascii && gh pr create --title "Public-release shine: README rewrite + per-area docs + GIF assets" --body "$(cat <<'EOF'
## Summary
- Compact (~150 line) main README with hero GIF, 4×3 env gallery, embedded tool GIFs, badges, quick-nav, and a documentation map.
- New per-area docs: docs/OBSERVATION_FORMAT.md, docs/INTEGRATION.md, docs/ARCHITECTURE.md, scripts/README.md.
- Refreshed: docs/REPLAY.md (now embeds gb_replay.gif), eval/README.md, src/glyphbench/rl/README.md.
- All 300 random-agent GIFs regenerated with a new colored Unicode glyph palette and re-uploaded to HF dataset `anon-paper-submission/glyphbench-assets`.
- Three new HF-hosted README assets: hero composite, demo_all_envs walkthrough, gb_replay walkthrough.

## Test plan
- [ ] Open the README on github.com/<repo>/blob/<branch>/README.md and confirm hero, gallery, and tool GIFs all load.
- [ ] Click through every link in the documentation map to confirm they resolve.
- [ ] Confirm leaderboard `gallery.html` (which pulls from the same HF location) automatically shows the new colored GIFs.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Surface the PR URL to the user.

- [ ] **Step 6: Done**

---

## Self-review notes (filled inline)

**Spec coverage:**
- Hero composite GIF → Task 7 ✓
- Per-suite gallery (12 envs) → Task 6 (renders) + Task 18 (gallery table) ✓
- gb replay walkthrough GIF → Task 9 ✓ (env switched to `minigrid-empty-5x5-v0`; documented at top)
- demo walkthrough GIF → Task 8 ✓
- 300 grid-only GIFs regen → Task 6 ✓
- HF upload → Task 10 ✓
- Glyph color palette → Task 4 (PALETTE in render_grid_gif.py) ✓
- Validation gate → Task 5 ✓
- README freshness audit → Task 2 ✓
- Subdir docs (OBSERVATION_FORMAT, INTEGRATION, ARCHITECTURE, scripts/README, REPLAY refresh, eval/README refresh, rl/README refresh) → Tasks 11-17 ✓
- Main README rewrite → Task 18 ✓
- Public-release readiness check → Task 19 ✓
- Cleanup → Task 20 ✓
- User checkpoints (audit judgment-calls, validation gate, branch view) → Tasks 2/5/19 explicitly pause ✓

**Type / signature consistency:**
- `render_env_gif(env_id, out_path, *, font_size, seed, duration_ms)` defined in Task 4, called only from `main()` in same file. ✓
- `_extract_grid` defined in Task 4, used only locally. ✓
- `_load_gif`, `_resize_to`, `_add_caption`, `_take_evenly`, `_crossfade` all defined and used in Task 7's build_hero.py. ✓
- HF URL pattern `https://huggingface.co/datasets/anon-paper-submission/glyphbench-assets/resolve/main/<path>` used consistently in Tasks 10/15/18/19. ✓
- Branch name `readme/public-release-shine` used in Tasks 19/20. ✓

**Placeholder scan:**
- Task 11 Step 3 has `[Document current behavior...]` — explicitly flagged as audit-driven; instruction says "Fill in all bracketed `[...]` placeholders before writing." ✓
- Task 16 Step 3 has `[Pull from run_debug.sh ...]` — explicitly audit-driven, same. ✓
- Task 17 Step 3 has `[Adjust file list ...]` — explicitly audit-driven. ✓
- Task 18 Step 2 has `[N]` and `[A]` and `[Replace ...]` — explicitly tied to Task 2 verified values. ✓
- These bracketed placeholders are NOT plan failures because each is paired with an explicit instruction in the same step describing exactly which prior task's verified output fills it.
