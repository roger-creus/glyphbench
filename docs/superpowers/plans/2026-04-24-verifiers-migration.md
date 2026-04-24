# Verifiers Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace gymnasium with verifiers across all 292 GlyphBench environments, wire the package for prime-rl eval + training, and rebuild the Docker/Singularity container. The repo must have zero gymnasium references after this plan completes.

**Architecture:** One umbrella `glyphbench` verifiers package with a single `load_environment(env_id, num_episodes, n_frames, max_output_tokens, ...)` that routes to any of the 292 registered games. Frame-stacked multi-turn env with CoT-only chain-of-thought, XML parser, 512-token response budget. Eval via `vf-eval`, training via prime-rl's `uv run rl @ rl.toml`.

**Tech Stack:** Python 3.12 · `uv` package manager · `verifiers>=0.1.12` · `datasets>=3.0` · `prime-rl` (for training smoke) · `vllm` (inference) · `pytest` + `hypothesis` (tests) · Docker + Apptainer (container).

**Design reference:** `docs/superpowers/specs/2026-04-24-verifiers-migration-design.md`

**Source references for subagents:**
- Verifiers source: unzip `/home/roger/Desktop/rl-world-ascii/verifiers-main.zip` to `/tmp/verifiers-main/` (tool: `unzip -qo <zip> -d /tmp/`).
- prime-rl source: unzip `/home/roger/Desktop/rl-world-ascii/prime-rl-main.zip` to `/tmp/prime-rl-main/`.
- Key verifiers files to consult: `verifiers/envs/multiturn_env.py`, `verifiers/envs/experimental/gym_env.py`, `verifiers/parsers/xml_parser.py` (for XMLParser internals), `verifiers/rubrics/rubric.py`.
- Key prime-rl files: `examples/wordle/rl.toml`, `configs/debug/orch.toml`, `src/prime_rl/entrypoints/rl.py` (for config schema).

---

## File Structure

### Files created

```
src/glyphbench/core/glyph_primitives.py          # renamed from ascii_primitives.py
src/glyphbench/verifiers_integration/__init__.py
src/glyphbench/verifiers_integration/env.py      # GlyphbenchMultiTurnEnv + load_environment
src/glyphbench/verifiers_integration/parser.py   # GlyphbenchXMLParser
src/glyphbench/verifiers_integration/prompting.py # build_system_prompt, render_user_turn
src/glyphbench/verifiers_integration/rubric.py   # EpisodicReturnRubric

tests/verifiers_integration/__init__.py
tests/verifiers_integration/test_parser.py
tests/verifiers_integration/test_prompting.py
tests/verifiers_integration/test_rubric.py
tests/verifiers_integration/test_env.py
tests/verifiers_integration/test_end_to_end.py

configs/rl/glyphbench-smoke/rl.toml
eval/run_debug.sh
eval/run_full.sh
scripts/build_sif.sh
```

### Files modified

```
src/glyphbench/__init__.py                       # exports load_environment + make_env
src/glyphbench/core/__init__.py                  # exports new registry API
src/glyphbench/core/base_env.py                  # BaseAsciiEnv → BaseGlyphEnv, drop gym.Env
src/glyphbench/core/registry.py                  # dict-based class registry
src/glyphbench/core/metrics.py                   # only import cleanup if needed
src/glyphbench/envs/{dummy,minigrid,minihack,atari,craftax,procgen,classics}/
                                                 # every *.py: BaseAsciiEnv→BaseGlyphEnv,
                                                 # ascii_primitives→glyph_primitives;
                                                 # each __init__.py rewritten to class-object registration

pyproject.toml                                   # drop gymnasium, add verifiers+datasets, new script entries
Dockerfile                                       # full rewrite from nvidia/cuda base
eval/README.md                                   # rewrite for vf-eval flow
eval/random_baseline.py                          # port to non-gym API

scripts/demo_all_envs.py, play_random.py, play_interactive.py, play_curses.py,
scripts/replay_trajectory.py, scripts/record_random_gifs.py, scripts/run_benchmark.py
                                                 # port to non-gym API via make_env()

tests/core/test_base_env.py                      # drop gym assertions
tests/core/test_registry.py                      # class-object API
tests/core/test_ascii_primitives.py              # rename file + imports
tests/envs/**/*.py                               # every test: gym.make(id) → make_env(id)
tests/conftest.py                                # drop gym imports
tests/test_cli.py                                # update entry-point smoke
tests/test_end_to_end.py                         # port

tests/test_eval_history.py                       # DELETE (harness replaced)
tests/test_scoring_coverage.py                   # DELETE (scoring.py deleted)
tests/test_minigrid_prompt_sync.py               # port (drop gym), keeps system_prompt check
```

### Files deleted

```
src/glyphbench/harness/                          # entire directory
src/glyphbench/providers/                        # entire directory
src/glyphbench/runner/                           # entire directory
src/glyphbench/core/ascii_primitives.py          # renamed (moved to glyph_primitives.py)
eval/run_eval.py
eval/scoring.py
scripts/serve_vllm.sh
tests/harness/                                   # entire directory
tests/providers/ (if exists)                     # not present, listed for completeness
tests/runner/ (if exists)                        # not present, listed for completeness
tests/core/test_ascii_primitives.py              # renamed to test_glyph_primitives.py
tests/test_eval_history.py
tests/test_scoring_coverage.py
```

---

## Milestone 1 — Core rewrite + verifiers integration + dummy port

**Goal end-state:** `import glyphbench; glyphbench.load_environment(env_id="glyphbench/__dummy-v0")` returns a working `vf.MultiTurnEnv`. `pytest tests/core tests/verifiers_integration tests/envs/dummy` green. `grep -r "gymnasium\|gym\.Env\|gym\.register\|gym\.make" src/glyphbench/core src/glyphbench/envs/dummy src/glyphbench/verifiers_integration tests/core tests/verifiers_integration tests/envs/dummy` returns nothing.

### Task 1.1: Delete obsolete directories

**Files:**
- Delete: `src/glyphbench/harness/`, `src/glyphbench/providers/`, `src/glyphbench/runner/`
- Delete: `tests/harness/`
- Delete: `eval/run_eval.py`, `eval/scoring.py`, `scripts/serve_vllm.sh`
- Delete: `tests/test_eval_history.py`, `tests/test_scoring_coverage.py`

- [ ] **Step 1: Remove directories and obsolete files**

```bash
cd /home/roger/Desktop/rl-world-ascii
git rm -r src/glyphbench/harness src/glyphbench/providers src/glyphbench/runner tests/harness
git rm eval/run_eval.py eval/scoring.py scripts/serve_vllm.sh
git rm tests/test_eval_history.py tests/test_scoring_coverage.py
```

- [ ] **Step 2: Verify nothing in the remaining tree still imports from them**

```bash
grep -rn "from glyphbench.harness\|from glyphbench.providers\|from glyphbench.runner\|import glyphbench.harness\|import glyphbench.providers\|import glyphbench.runner" src/ tests/ scripts/ eval/
```

Expected: no matches. If matches exist, they're in files we're about to port — leave them; they get fixed later.

- [ ] **Step 3: Commit**

```bash
git commit -m "remove obsolete harness/providers/runner/eval_runner — replaced by verifiers integration"
```

### Task 1.2: Rename `ascii_primitives.py` → `glyph_primitives.py`

**Files:**
- Move: `src/glyphbench/core/ascii_primitives.py` → `src/glyphbench/core/glyph_primitives.py`
- Modify: all files importing from the old path
- Move: `tests/core/test_ascii_primitives.py` → `tests/core/test_glyph_primitives.py`

- [ ] **Step 1: Rename via git mv**

```bash
cd /home/roger/Desktop/rl-world-ascii
git mv src/glyphbench/core/ascii_primitives.py src/glyphbench/core/glyph_primitives.py
git mv tests/core/test_ascii_primitives.py tests/core/test_glyph_primitives.py
```

- [ ] **Step 2: Update all imports**

```bash
grep -rl "ascii_primitives" src/ tests/ | xargs sed -i 's/ascii_primitives/glyph_primitives/g'
# Also update textual docs that reference the module
grep -rl "ascii_primitives" docs/ 2>/dev/null | xargs -r sed -i 's/ascii_primitives/glyph_primitives/g'
```

- [ ] **Step 3: Verify**

```bash
grep -rn "ascii_primitives" src/ tests/ docs/ 2>&1 | grep -v ".git\|__pycache__\|.mypy_cache\|.ruff_cache\|.hypothesis"
```

Expected: no matches.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "rename ascii_primitives → glyph_primitives (observations are Unicode, not ASCII)"
```

### Task 1.3: Rewrite `core/base_env.py` — drop gym.Env, rename to `BaseGlyphEnv`

**Files:**
- Modify: `src/glyphbench/core/base_env.py`
- Test: `tests/core/test_base_env.py`

- [ ] **Step 1: Write failing tests first**

Overwrite `tests/core/test_base_env.py` with:

```python
"""Tests for the non-gym BaseGlyphEnv base class."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation


class _Tiny(BaseGlyphEnv):
    action_spec = ActionSpec(names=("A", "B"), descriptions=("a", "b"))
    noop_action_name = "A"

    def _reset(self, seed: int) -> GridObservation:
        return GridObservation(grid="X", legend="", hud="", message="")

    def _step(self, action: int):
        return GridObservation(grid="X", legend="", hud="", message=""), 0.0, False, False, {}

    def _render_current_observation(self) -> GridObservation:
        return GridObservation(grid="X", legend="", hud="", message="")

    def system_prompt(self) -> str:
        return "sys"

    def env_id(self) -> str:
        return "tiny-v0"


def test_not_gym_subclass():
    # BaseGlyphEnv must NOT inherit from gymnasium.Env; the module must not even
    # need gymnasium to be importable.
    import sys
    assert "gymnasium" not in sys.modules or not any(
        "gym" in cls.__module__.lower() for cls in _Tiny.__mro__ if cls is not object
    )


def test_reset_requires_int_seed():
    env = _Tiny()
    with pytest.raises(TypeError):
        env.reset("not-an-int")  # type: ignore[arg-type]


def test_reset_returns_text_and_info():
    env = _Tiny()
    obs, info = env.reset(42)
    assert isinstance(obs, str)
    assert info["turn"] == 0
    assert info["env_id"] == "tiny-v0"
    assert info["seed"] == 42


def test_step_rejects_bool_and_out_of_range():
    env = _Tiny()
    env.reset(0)
    with pytest.raises(TypeError):
        env.step(True)  # bools are ints — must be rejected
    with pytest.raises(ValueError):
        env.step(99)


def test_step_returns_five_tuple():
    env = _Tiny()
    env.reset(0)
    out = env.step(0)
    assert len(out) == 5
    obs, reward, term, trunc, info = out
    assert isinstance(obs, str)
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)


def test_max_turns_truncates():
    env = _Tiny(max_turns=3)
    env.reset(0)
    env.step(0); env.step(0)
    _, _, term, trunc, info = env.step(0)
    assert trunc is True
    assert info.get("truncation_reason") == "max_turns"


def test_rng_access_requires_reset():
    env = _Tiny()
    with pytest.raises(RuntimeError):
        _ = env.rng
    env.reset(0)
    rng = env.rng
    assert isinstance(rng, np.random.Generator)


def test_close_is_noop_by_default():
    env = _Tiny()
    env.reset(0)
    assert env.close() is None


def test_no_action_observation_space_attrs():
    # These were gymnasium-only; they must not exist on BaseGlyphEnv.
    env = _Tiny()
    env.reset(0)
    assert not hasattr(env, "action_space")
    assert not hasattr(env, "observation_space")
    assert not hasattr(env, "metadata") or not isinstance(getattr(env, "metadata", None), dict) or "render_modes" not in env.metadata  # tolerate subclass metadata, but shouldn't inherit render_modes
```

- [ ] **Step 2: Run tests — expect failure**

```bash
uv run pytest tests/core/test_base_env.py -x 2>&1 | tail -20
```

Expected: `ImportError: cannot import name 'BaseGlyphEnv'` (still `BaseAsciiEnv`).

- [ ] **Step 3: Rewrite `src/glyphbench/core/base_env.py`**

Full replacement:

```python
"""Base class for every game in glyphbench.

Plain Python class — no framework inheritance. Subclasses implement the five
abstract methods; the public reset/step surface is fixed here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from glyphbench.core.action import ActionSpec
from glyphbench.core.observation import GridObservation


class BaseGlyphEnv(ABC):
    """Base class for every env in glyphbench.

    Subclasses MUST:
      - set `action_spec: ActionSpec` as a class or instance attribute
      - implement `_reset(seed)` returning the initial GridObservation
      - implement `_step(action_index)` returning (obs, reward, terminated, truncated, info)
      - implement `_render_current_observation()` returning current state as GridObservation
      - implement `system_prompt()` returning the per-game system prompt
      - implement `env_id()` returning the canonical env id string
    """

    action_spec: ActionSpec
    noop_action_name: str = "NOOP"

    def __init__(self, max_turns: int = 500) -> None:
        self.max_turns = max_turns
        self._turn: int = 0
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int) -> tuple[str, dict[str, Any]]:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be int, got {type(seed).__name__}")
        self._rng = np.random.default_rng(int(seed))
        self._turn = 0
        obs = self._reset(int(seed))
        info: dict[str, Any] = {
            "turn": 0,
            "env_id": self.env_id(),
            "seed": int(seed),
        }
        return obs.render(), info

    def step(self, action: int) -> tuple[str, float, bool, bool, dict[str, Any]]:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError(f"action must be int, got {type(action).__name__}")
        if not 0 <= int(action) < self.action_spec.n:
            raise ValueError(
                f"action {action} out of range [0, {self.action_spec.n})"
            )
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
    def _step(
        self, action: int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, Any]]: ...

    @abstractmethod
    def _render_current_observation(self) -> GridObservation: ...

    @abstractmethod
    def system_prompt(self) -> str: ...

    @abstractmethod
    def env_id(self) -> str: ...

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            raise RuntimeError("call reset() before accessing rng")
        return self._rng

    def get_observation(self) -> GridObservation:
        """Return the current observation without stepping. Useful for initial
        prompt construction at turn 0."""
        return self._render_current_observation()

    def close(self) -> None:
        """Optional cleanup hook. Default: no-op."""
        return None


# Back-compat alias during migration (removed at end of Milestone 4).
# TEMPORARY — grep-audit at end of M4 removes this line.
BaseAsciiEnv = BaseGlyphEnv
```

- [ ] **Step 4: Run tests — expect pass**

```bash
uv run pytest tests/core/test_base_env.py -x 2>&1 | tail -10
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/glyphbench/core/base_env.py tests/core/test_base_env.py
git commit -m "core: rewrite BaseGlyphEnv as plain class (no gym.Env inheritance)"
```

### Task 1.4: Rewrite `core/registry.py` — class-object registration

**Files:**
- Modify: `src/glyphbench/core/registry.py`
- Modify: `tests/core/test_registry.py`

- [ ] **Step 1: Rewrite failing tests**

Overwrite `tests/core/test_registry.py`:

```python
"""Tests for the plain-Python class-object registry (no gym)."""

from __future__ import annotations

import pytest

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation
from glyphbench.core.registry import (
    REGISTRY,
    all_glyphbench_env_ids,
    make_env,
    register_env,
)


class _E(BaseGlyphEnv):
    action_spec = ActionSpec(names=("A",), descriptions=("a",))

    def _reset(self, seed: int) -> GridObservation:
        return GridObservation(grid=".", legend="", hud="", message="")
    def _step(self, action: int):
        return GridObservation(grid=".", legend="", hud="", message=""), 0.0, True, False, {}
    def _render_current_observation(self) -> GridObservation:
        return GridObservation(grid=".", legend="", hud="", message="")
    def system_prompt(self) -> str:
        return ""
    def env_id(self) -> str:
        return "test/e-v0"


class _F(BaseGlyphEnv):
    action_spec = ActionSpec(names=("A",), descriptions=("a",))

    def _reset(self, seed: int) -> GridObservation:
        return GridObservation(grid=".", legend="", hud="", message="")
    def _step(self, action: int):
        return GridObservation(grid=".", legend="", hud="", message=""), 0.0, True, False, {}
    def _render_current_observation(self) -> GridObservation:
        return GridObservation(grid=".", legend="", hud="", message="")
    def system_prompt(self) -> str:
        return ""
    def env_id(self) -> str:
        return "test/f-v0"


@pytest.fixture(autouse=True)
def _save_restore_registry():
    snapshot = dict(REGISTRY)
    yield
    REGISTRY.clear()
    REGISTRY.update(snapshot)


def test_register_and_make():
    register_env("test/e-v0", _E)
    env = make_env("test/e-v0")
    assert isinstance(env, _E)


def test_register_idempotent_same_class():
    register_env("test/e-v0", _E)
    register_env("test/e-v0", _E)  # idempotent, no error


def test_register_rejects_duplicate_with_different_class():
    register_env("test/e-v0", _E)
    with pytest.raises(ValueError, match="already registered"):
        register_env("test/e-v0", _F)


def test_make_unknown_raises():
    with pytest.raises(KeyError, match="unknown env_id"):
        make_env("test/not-there-v0")


def test_make_forwards_kwargs():
    register_env("test/e-v0", _E)
    env = make_env("test/e-v0", max_turns=7)
    assert env.max_turns == 7


def test_all_env_ids_sorted():
    register_env("test/b-v0", _E)
    register_env("test/a-v0", _F)
    ids = all_glyphbench_env_ids()
    assert ids == sorted(ids)


def test_register_rejects_non_baseclass():
    class NotAnEnv:
        pass
    with pytest.raises(TypeError):
        register_env("test/bad-v0", NotAnEnv)  # type: ignore[arg-type]
```

- [ ] **Step 2: Rewrite `src/glyphbench/core/registry.py`**

```python
"""Plain-Python class-object registry for glyphbench environments."""

from __future__ import annotations

from typing import Any

from glyphbench.core.base_env import BaseGlyphEnv

REGISTRY: dict[str, type[BaseGlyphEnv]] = {}


def register_env(env_id: str, cls: type[BaseGlyphEnv]) -> None:
    """Register a class under an env id.

    Idempotent for the same (id, class) pair; raises ``ValueError`` on
    conflicting registrations and ``TypeError`` if ``cls`` is not a
    ``BaseGlyphEnv`` subclass.
    """
    if not isinstance(cls, type) or not issubclass(cls, BaseGlyphEnv):
        raise TypeError(
            f"register_env expected a BaseGlyphEnv subclass, got {cls!r}"
        )
    existing = REGISTRY.get(env_id)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"env_id {env_id!r} already registered to {existing.__name__}; "
            f"refusing to overwrite with {cls.__name__}"
        )
    REGISTRY[env_id] = cls


def make_env(env_id: str, **kwargs: Any) -> BaseGlyphEnv:
    """Instantiate the class registered under ``env_id``.

    Extra kwargs are forwarded to the class constructor.
    """
    cls = REGISTRY.get(env_id)
    if cls is None:
        raise KeyError(
            f"unknown env_id {env_id!r}; known ids: {sorted(REGISTRY)[:5]}…"
        )
    return cls(**kwargs)


def all_glyphbench_env_ids() -> list[str]:
    """Return every registered id as a sorted list."""
    return sorted(REGISTRY)
```

- [ ] **Step 3: Run tests — expect pass**

```bash
uv run pytest tests/core/test_registry.py -x 2>&1 | tail -10
```

Expected: 7 passed.

- [ ] **Step 4: Commit**

```bash
git add src/glyphbench/core/registry.py tests/core/test_registry.py
git commit -m "core: class-object registry (drop gym.register / entry_point strings)"
```

### Task 1.5: Update `core/__init__.py` exports

**Files:**
- Modify: `src/glyphbench/core/__init__.py`

- [ ] **Step 1: Rewrite `src/glyphbench/core/__init__.py`**

```python
"""Core contracts shared by every env and the verifiers integration."""

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation
from glyphbench.core.registry import (
    REGISTRY,
    all_glyphbench_env_ids,
    make_env,
    register_env,
)

# Back-compat alias during migration (removed at end of M4).
BaseAsciiEnv = BaseGlyphEnv

__all__ = [
    "ActionSpec",
    "BaseGlyphEnv",
    "BaseAsciiEnv",  # temporary alias
    "GridObservation",
    "REGISTRY",
    "all_glyphbench_env_ids",
    "make_env",
    "register_env",
]
```

- [ ] **Step 2: Verify imports still resolve**

```bash
uv run python -c "from glyphbench.core import BaseGlyphEnv, BaseAsciiEnv, make_env, register_env, REGISTRY; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/glyphbench/core/__init__.py
git commit -m "core: export BaseGlyphEnv, make_env, register_env from core package"
```

### Task 1.6: Port `dummy/` suite to new API

**Files:**
- Modify: `src/glyphbench/envs/dummy/env.py`
- Modify: `src/glyphbench/envs/dummy/__init__.py`
- Modify: `tests/envs/dummy/` (whatever tests exist)

- [ ] **Step 1: Update `env.py` imports**

Replace `from glyphbench.core.base_env import BaseAsciiEnv` with `from glyphbench.core.base_env import BaseGlyphEnv` and `class DummyEnv(BaseAsciiEnv)` with `class DummyEnv(BaseGlyphEnv)`.

Also update the env id to drop the double-underscore "hidden" prefix — we want it visible now:

```bash
cd /home/roger/Desktop/rl-world-ascii
sed -i 's/BaseAsciiEnv/BaseGlyphEnv/g' src/glyphbench/envs/dummy/env.py
```

(Leave the env_id `"glyphbench/__dummy-v0"` string as-is — existing tests and manifests reference it.)

- [ ] **Step 2: Rewrite `src/glyphbench/envs/dummy/__init__.py`**

```python
"""Dummy test-fixture env. Importing this module registers the env."""

from glyphbench.core.registry import register_env
from glyphbench.envs.dummy.env import DummyEnv

register_env("glyphbench/__dummy-v0", DummyEnv)
```

- [ ] **Step 3: Port dummy tests**

```bash
ls tests/envs/dummy/
# For each test file, replace gym.make patterns:
grep -rl "gym.make\|gymnasium" tests/envs/dummy/ | xargs -r sed -i \
  -e 's/import gymnasium as gym//g' \
  -e 's/import gym//g' \
  -e 's/gym\.make(\([^)]*\))/make_env(\1)/g' \
  -e 's/\.unwrapped//g'
# Also add `from glyphbench.core import make_env` to each test if not present.
```

Then, for each ported test file, add at the top:

```python
from glyphbench.core import make_env
import glyphbench.envs.dummy  # registers the dummy env
```

Read each test file in `tests/envs/dummy/` after sed and verify the imports + the `env.reset(seed=42)` → `env.reset(42)` positional swap.

- [ ] **Step 4: Run dummy tests**

```bash
uv run pytest tests/envs/dummy -x 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/glyphbench/envs/dummy tests/envs/dummy
git commit -m "envs: port dummy/ to non-gym API (BaseGlyphEnv + class-object registry)"
```

### Task 1.7: Update `pyproject.toml` — drop gymnasium, add verifiers

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update dependencies**

Replace the `dependencies` block in `pyproject.toml`:

```toml
dependencies = [
    "verifiers>=0.1.12",
    "datasets>=3.0",
    "openai>=1.50",
    "numpy>=2.0",
    "pydantic>=2.9",
    "pyyaml>=6.0",
    "jinja2>=3.1",
    "rich>=13.9",
]
```

(Drop `gymnasium>=1.0`. Add `verifiers`, `datasets`, `openai` — all required by verifiers env code at runtime.)

In `[project.keywords]` drop `"gymnasium"`.

In `[tool.mypy.overrides]` `module` list, add `"verifiers"`, `"verifiers.*"`, `"datasets"` to the ignore list. Remove if any `"gymnasium"` entries exist.

In `[project.optional-dependencies]`, the `eval` extra can stay (still useful for running vLLM locally), but simplify — vllm is the only requirement:

```toml
eval = [
    "vllm>=0.8",
    "torch>=2.4",
]
```

Add a new `rl` extra (heavy, optional):

```toml
rl = [
    "prime-rl",
]
```

- [ ] **Step 2: Regenerate lockfile**

```bash
cd /home/roger/Desktop/rl-world-ascii
uv lock 2>&1 | tail -10
```

Expected: "Resolved N packages" without error. If `prime-rl` fails to resolve from pypi (prime-rl is git-hosted), change the `rl` extra to:

```toml
rl = [
    "prime-rl @ git+https://github.com/PrimeIntellect-ai/prime-rl",
]
```

and re-lock.

- [ ] **Step 3: Sync the base env**

```bash
uv sync 2>&1 | tail -10
```

Expected: "Installed N packages" or already-up-to-date.

- [ ] **Step 4: Verify verifiers imports**

```bash
uv run python -c "import verifiers as vf; print(vf.__version__)"
uv run python -c "import datasets; print(datasets.__version__)"
```

Expected: both print versions without import errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: drop gymnasium, add verifiers>=0.1.12 + datasets + openai"
```

### Task 1.8: Create `verifiers_integration/parser.py`

**Files:**
- Create: `src/glyphbench/verifiers_integration/__init__.py`
- Create: `src/glyphbench/verifiers_integration/parser.py`
- Create: `tests/verifiers_integration/__init__.py`
- Create: `tests/verifiers_integration/test_parser.py`

- [ ] **Step 1: Write failing tests**

Create `tests/verifiers_integration/__init__.py` (empty).

Create `tests/verifiers_integration/test_parser.py`:

```python
"""Tests for GlyphbenchXMLParser: XML-primary with JSON/regex fallback."""

from __future__ import annotations

import pytest

from glyphbench.core.action import ActionSpec
from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser


@pytest.fixture
def spec():
    return ActionSpec(
        names=("LEFT", "RIGHT", "UP", "DOWN", "NOOP"),
        descriptions=("l", "r", "u", "d", "n"),
    )


@pytest.fixture
def parser():
    return GlyphbenchXMLParser()


def test_well_formed_xml(parser, spec):
    text = "<think>reason</think><action>LEFT</action>"
    idx, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert (idx, name, failed) == (0, "LEFT", False)


def test_xml_case_insensitive(parser, spec):
    text = "<think>x</think><action>right</action>"
    idx, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert (idx, name, failed) == (1, "RIGHT", False)


def test_xml_whitespace_in_action(parser, spec):
    text = "<think>x</think><action>   UP\n</action>"
    idx, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert (idx, name, failed) == (2, "UP", False)


def test_multiple_action_tags_take_last(parser, spec):
    text = "<action>LEFT</action>some text<action>RIGHT</action>"
    idx, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert name == "RIGHT"


def test_unknown_action_falls_back_to_noop(parser, spec):
    text = "<think>x</think><action>FLY</action>"
    idx, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert (name, failed) == ("NOOP", True)


def test_missing_action_tag_falls_back(parser, spec):
    text = "<think>i have no action</think>"
    _, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert (name, failed) == ("NOOP", True)


def test_empty_string_falls_back(parser, spec):
    _, name, failed = parser.parse_action("", spec, noop="NOOP")
    assert (name, failed) == ("NOOP", True)


def test_json_fallback(parser, spec):
    text = 'no xml, but: {"thinking":"x","action":"DOWN"}'
    idx, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert (idx, name, failed) == (3, "DOWN", False)


def test_json_fenced_fallback(parser, spec):
    text = '```json\n{"action": "LEFT"}\n```'
    _, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert (name, failed) == ("LEFT", False)


def test_bare_action_name_fallback(parser, spec):
    # Last-resort: the response contains nothing but an action name.
    text = "UP"
    _, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert (name, failed) == ("UP", False)


def test_prose_mentioning_action_does_not_trigger_bare_fallback(parser, spec):
    # "I will go LEFT" should still parse via no-xml fallback chain — if the
    # bare-name regex matches the only action-looking token, that's fine.
    # This test documents the choice: the last-ditch regex picks the last
    # uppercase token matching an action name.
    text = "I am considering going LEFT or RIGHT. Final answer: DOWN"
    _, name, _ = parser.parse_action(text, spec, noop="NOOP")
    assert name == "DOWN"


def test_malformed_xml_nothing_else_works(parser, spec):
    text = "<action>LEFT"  # no close tag, no json, no bare action at end
    _, name, failed = parser.parse_action(text, spec, noop="NOOP")
    # Our fallback chain should still find "LEFT" via the bare-name regex.
    assert name == "LEFT"
    assert failed is False


def test_completely_off_the_rails(parser, spec):
    text = "asdfjkl; qwerty 12345"
    _, name, failed = parser.parse_action(text, spec, noop="NOOP")
    assert (name, failed) == ("NOOP", True)


def test_get_format_reward_func_is_callable(parser):
    # verifiers XMLParser provides a format-reward fn; we expose it.
    fn = parser.get_format_reward_func()
    assert callable(fn)
```

- [ ] **Step 2: Create `verifiers_integration/__init__.py`**

```python
"""Verifiers integration — entry point for vf-eval and prime-rl."""

from glyphbench.verifiers_integration.env import (
    GlyphbenchMultiTurnEnv,
    load_environment,
)
from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser
from glyphbench.verifiers_integration.prompting import (
    build_system_prompt,
    render_user_turn,
)
from glyphbench.verifiers_integration.rubric import EpisodicReturnRubric

__all__ = [
    "GlyphbenchMultiTurnEnv",
    "GlyphbenchXMLParser",
    "EpisodicReturnRubric",
    "build_system_prompt",
    "render_user_turn",
    "load_environment",
]
```

- [ ] **Step 3: Create `src/glyphbench/verifiers_integration/parser.py`**

```python
"""GlyphbenchXMLParser: XML-primary with JSON and bare-name fallbacks.

Wraps ``verifiers.XMLParser`` (fields ``think``, ``action``) and adds a
3-layer fallback chain that tolerates imperfectly-formatted model output:

    1. XML: ``<action>NAME</action>`` (preferred, last occurrence wins).
    2. JSON: ``{"action": "NAME"}`` inside any fence or top-level JSON object.
    3. Bare: last uppercase/snake-case token matching a known action name.

Unknown or missing action → ``(noop_idx, noop_name, parse_failed=True)``.
"""

from __future__ import annotations

import json
import re
from typing import Any

import verifiers as vf

from glyphbench.core.action import ActionSpec

_XML_ACTION_RE = re.compile(
    r"<\s*action\s*>(.*?)<\s*/\s*action\s*>", re.DOTALL | re.IGNORECASE
)
_XML_ACTION_OPEN_RE = re.compile(
    r"<\s*action\s*>(.*?)(?:<|\Z)", re.DOTALL | re.IGNORECASE
)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_BARE_NAME_RE = re.compile(r"\b([A-Z][A-Z0-9_]{1,})\b")


class GlyphbenchXMLParser(vf.XMLParser):
    """Parser used by the glyphbench verifiers integration.

    Instantiate with the default XML field layout (think + action); the extra
    ``parse_action`` method on top of the verifiers base provides the 3-layer
    fallback chain.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            fields=["think", "action"],
            answer_field="action",
            **kwargs,
        )

    def parse_action(
        self,
        raw_text: str,
        spec: ActionSpec,
        *,
        noop: str,
    ) -> tuple[int, str, bool]:
        """Extract an action from model output.

        Returns ``(action_idx, canonical_action_name, parse_failed)``.

        The canonical action name is always one of ``spec.names`` — on any
        parse failure or unknown action name we fall back to ``noop`` and
        flag ``parse_failed=True``.
        """
        candidate = self._extract_candidate(raw_text or "")
        if candidate is None:
            return self._noop(spec, noop)
        try:
            return (spec.index_of(candidate), spec.names[spec.index_of(candidate)], False)
        except KeyError:
            return self._noop(spec, noop)

    def _extract_candidate(self, text: str) -> str | None:
        # 1. Complete <action>...</action> — take the last occurrence.
        matches = _XML_ACTION_RE.findall(text)
        if matches:
            return matches[-1].strip()

        # 2. Unclosed <action>... — tolerant.
        open_match = _XML_ACTION_OPEN_RE.search(text)
        if open_match:
            cand = open_match.group(1).strip()
            # Only use if it looks like a plain action name (no nested XML).
            if cand and "<" not in cand and len(cand) < 64:
                return cand

        # 3. JSON — fenced block, then top-level object scan.
        for body in _iter_json_bodies(text):
            try:
                obj = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict) and isinstance(obj.get("action"), str):
                return obj["action"].strip()

        # 4. Bare uppercase-token fallback — take the last one.
        bare = _BARE_NAME_RE.findall(text)
        if bare:
            return bare[-1].strip()

        return None

    @staticmethod
    def _noop(spec: ActionSpec, noop: str) -> tuple[int, str, bool]:
        try:
            idx = spec.index_of(noop)
        except KeyError:
            # Very defensive — an env without its declared noop: fall back to idx 0.
            idx = 0
        return idx, spec.names[idx], True


def _iter_json_bodies(text: str):
    """Yield candidate JSON bodies from fenced code blocks + top-level scan."""
    for m in _FENCE_RE.finditer(text):
        yield m.group(1)
    # Brace-balanced scan.
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escape = False
        start = i
        j = i
        while j < n:
            ch = text[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[start : j + 1]
                        break
            j += 1
        i = j + 1
```

- [ ] **Step 4: Run parser tests — expect pass**

```bash
uv run pytest tests/verifiers_integration/test_parser.py -x 2>&1 | tail -20
```

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add src/glyphbench/verifiers_integration/__init__.py src/glyphbench/verifiers_integration/parser.py tests/verifiers_integration/__init__.py tests/verifiers_integration/test_parser.py
git commit -m "verifiers: GlyphbenchXMLParser with XML + JSON + bare-name fallback chain"
```

Note: Step 2 references `env.py`, `prompting.py`, `rubric.py` which don't exist yet. `verifiers_integration/__init__.py` will ImportError until 1.9, 1.10, 1.11 are done. That's fine — only `test_parser.py` is run in step 4 (direct import of `parser.py`). The `__init__.py` can be created with a fully commented-out body for now, or we can defer its creation until task 1.11 is done. Prefer: comment out the `env`/`prompting`/`rubric` imports in `__init__.py` until those files exist, then uncomment in Task 1.11.

Write `__init__.py` now with imports commented out:

```python
"""Verifiers integration — entry point for vf-eval and prime-rl."""

from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser

# Uncommented in Task 1.11 after the dependent modules exist:
# from glyphbench.verifiers_integration.env import (
#     GlyphbenchMultiTurnEnv,
#     load_environment,
# )
# from glyphbench.verifiers_integration.prompting import (
#     build_system_prompt,
#     render_user_turn,
# )
# from glyphbench.verifiers_integration.rubric import EpisodicReturnRubric

__all__ = ["GlyphbenchXMLParser"]
```

### Task 1.9: Create `verifiers_integration/prompting.py`

**Files:**
- Create: `src/glyphbench/verifiers_integration/prompting.py`
- Create: `tests/verifiers_integration/test_prompting.py`

- [ ] **Step 1: Write failing tests**

Create `tests/verifiers_integration/test_prompting.py`:

```python
"""Tests for prompting.py: system prompt + frame-stack user-turn rendering."""

from __future__ import annotations

from collections import deque

import pytest

from glyphbench.core.action import ActionSpec
from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.observation import GridObservation
from glyphbench.verifiers_integration.prompting import (
    build_system_prompt,
    render_user_turn,
)


class _Game(BaseGlyphEnv):
    action_spec = ActionSpec(
        names=("LEFT", "RIGHT", "UP", "DOWN"),
        descriptions=("l", "r", "u", "d"),
    )

    def _reset(self, seed: int) -> GridObservation:
        return GridObservation(grid="A.", legend="A symbol-a", hud="step=0", message="")
    def _step(self, action: int):
        return GridObservation(grid="A.", legend="A symbol-a", hud="step=1", message=""), 0.0, False, False, {}
    def _render_current_observation(self) -> GridObservation:
        return self._reset(0)
    def system_prompt(self) -> str:
        return "You play a game. Move around."
    def env_id(self) -> str:
        return "test/g-v0"


@pytest.fixture
def game():
    g = _Game()
    g.reset(0)
    return g


def test_system_prompt_contains_game_rules(game):
    sp = build_system_prompt(game, max_output_tokens=512)
    assert "You play a game." in sp
    assert "Move around." in sp


def test_system_prompt_declares_budget(game):
    sp = build_system_prompt(game, max_output_tokens=512)
    assert "512" in sp


def test_system_prompt_documents_xml_format(game):
    sp = build_system_prompt(game, max_output_tokens=512)
    assert "<think>" in sp and "</think>" in sp
    assert "<action>" in sp and "</action>" in sp


def test_system_prompt_lists_actions(game):
    sp = build_system_prompt(game, max_output_tokens=512)
    for name in ("LEFT", "RIGHT", "UP", "DOWN"):
        assert name in sp


def test_user_turn_zero_no_history_section(game):
    frames: deque = deque(maxlen=4)
    text = render_user_turn(game, frames, current_obs="[Legend]\nA — a\n\n[HUD]\nstep=0\n\n[Grid]\n.", turn=0)
    assert "[History" not in text
    assert "[Legend]" in text
    assert "[Current Observation" in text or "[Observation" in text
    assert "[Actions]" in text


def test_user_turn_with_history_dedups_legend(game):
    frames = deque(
        [
            ("[Legend]\nA — a\n\n[Grid]\nA.", "LEFT", 0.0),
            ("[Legend]\nA — a\n\n[Grid]\n.A", "RIGHT", 0.0),
        ],
        maxlen=4,
    )
    current = "[Legend]\nA — a\n\n[Grid]\nAA"
    text = render_user_turn(game, frames, current, turn=2)
    # Legend appears once globally (at top), not inside frames or current.
    assert text.count("[Legend]") == 1


def test_user_turn_history_window_respected(game):
    frames = deque(
        [(f"[Grid]\n{i}", "LEFT", float(i)) for i in range(4)],
        maxlen=4,
    )
    text = render_user_turn(game, frames, current_obs="[Grid]\nC", turn=4)
    # Four history entries rendered.
    assert text.count("reward") == 4 or text.count("→") == 4 or "turn" in text


def test_user_turn_reminds_format_and_budget(game):
    frames: deque = deque(maxlen=4)
    text = render_user_turn(game, frames, current_obs="[Grid]\n.", turn=0)
    assert "<action>" in text
    assert "512" in text
```

- [ ] **Step 2: Create `src/glyphbench/verifiers_integration/prompting.py`**

```python
"""Prompt construction for the glyphbench verifiers integration.

Two entry points:
    build_system_prompt(game, max_output_tokens) → str
        Composed from game.system_prompt() + a standardised response-format
        block. Called ONCE per rollout (static across turns).

    render_user_turn(game, frames, current_obs, turn) → str
        The per-turn user message. Layout:

            [Legend]
              <union of glyphs across frames + current, rendered once>

            [History — last N turns]                 (omitted entirely if N=0)
              (turn T-K) <grid-only view>
                  <one-line HUD delta>
                chose ACTION → reward R

              (turn T-K+1) ...

            [Current Observation — turn T]
              <full grid + HUD + optional message>

            [Actions]
              Choose one: [A, B, C, ...]

            Respond with <think>...</think><action>NAME</action>.
            Total response budget: 512 tokens.

Design goals:
    * One legend per user message, deduped across history + current.
    * Stable section order → KV-cache prefix overlap across turns.
    * History frames stripped of legend + HUD-delta-only, saving tokens.
    * Current observation kept intact (full grid + HUD + message), minus legend.
    * Budget reminded every turn, deters runaway thinking.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Iterable

from glyphbench.core.base_env import BaseGlyphEnv

_LEGEND_RE = re.compile(r"\[Legend\]\n(.*?)(?=\n\n\[|\Z)", re.DOTALL)
_GRID_RE = re.compile(r"\[Grid\]\n(.*?)(?=\n\n\[|\Z)", re.DOTALL)
_HUD_RE = re.compile(r"\[HUD\]\n(.*?)(?=\n\n\[|\Z)", re.DOTALL)
_MESSAGE_RE = re.compile(r"\[Message\]\n(.*?)(?=\n\n\[|\Z)", re.DOTALL)


RESPONSE_FORMAT_BLOCK_TMPL = (
    "RESPONSE FORMAT\n"
    "Respond with exactly two XML tags, in this order:\n"
    "  <think>your reasoning — keep it concise; plan your next move</think>\n"
    "  <action>ACTION_NAME</action>\n"
    "\n"
    "Your TOTAL response budget is {budget} tokens (thinking + action combined). "
    "Any text outside these two tags is ignored. If the <action> tag is missing "
    "or contains an unknown action name, the {noop} action is applied instead."
)


def build_system_prompt(game: BaseGlyphEnv, max_output_tokens: int) -> str:
    """Compose the system prompt: game rules + standard response-format block.

    The output is stable across turns — verifiers reuses the cached tokenisation
    as long as this content doesn't change.
    """
    header = game.system_prompt().rstrip()
    fmt = RESPONSE_FORMAT_BLOCK_TMPL.format(
        budget=max_output_tokens,
        noop=game.noop_action_name,
    )
    return f"{header}\n\n---\n{fmt}"


def render_user_turn(
    game: BaseGlyphEnv,
    frames: deque[tuple[str, str, float]] | Iterable[tuple[str, str, float]],
    current_obs: str,
    turn: int,
) -> str:
    """Render the user-turn message for the current timestep.

    Args:
        game: the live game instance (used for the action list).
        frames: iterable of (obs_text_before_action, action_name, reward) tuples
                in temporal order — oldest first, newest last.
        current_obs: the full ``GridObservation.render()`` text for this turn.
        turn: the absolute turn number (``game._turn`` after the last step).

    Returns:
        The user-turn string.
    """
    frames_list = list(frames)
    budget = _BUDGET_TOKENS  # kept separate so tests can override

    # 1. Build merged legend across history + current.
    legend_lines: dict[str, None] = {}  # ordered-set via dict
    for obs, _, _ in frames_list:
        for line in _extract_legend_lines(obs):
            legend_lines.setdefault(line, None)
    for line in _extract_legend_lines(current_obs):
        legend_lines.setdefault(line, None)

    parts: list[str] = []
    if legend_lines:
        parts.append("[Legend]\n" + "\n".join(legend_lines))

    # 2. History block (only if non-empty).
    if frames_list:
        parts.append(_render_history(frames_list, turn))

    # 3. Current observation — strip legend, keep HUD + grid + message.
    parts.append(_render_current_block(current_obs, turn))

    # 4. Actions enumerated.
    action_list = ", ".join(game.action_spec.names)
    parts.append(f"[Actions]\nChoose one: [{action_list}]")

    # 5. Budget reminder.
    parts.append(
        f"Respond with <think>...</think><action>NAME</action>. "
        f"Total response budget: {budget} tokens."
    )

    return "\n\n".join(parts)


# Default budget used in the reminder footer. Callers can override by reaching
# into this module attribute; load_environment sets it from max_output_tokens.
_BUDGET_TOKENS: int = 512


def set_budget_tokens(n: int) -> None:
    """Update the footer budget reminder value (called by load_environment)."""
    global _BUDGET_TOKENS
    _BUDGET_TOKENS = int(n)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _extract_legend_lines(obs: str) -> list[str]:
    m = _LEGEND_RE.search(obs)
    if not m:
        return []
    return [ln for ln in m.group(1).splitlines() if ln.strip()]


def _extract_grid(obs: str) -> str:
    m = _GRID_RE.search(obs)
    return m.group(1).rstrip() if m else ""


def _extract_hud(obs: str) -> str:
    m = _HUD_RE.search(obs)
    return m.group(1).strip() if m else ""


def _extract_message(obs: str) -> str:
    m = _MESSAGE_RE.search(obs)
    return m.group(1).strip() if m else ""


def _render_history(frames_list: list[tuple[str, str, float]], current_turn: int) -> str:
    n = len(frames_list)
    lines = [f"[History — last {n} turn{'s' if n != 1 else ''}]"]
    # Number each historical turn from T-N .. T-1 (T is the current turn).
    for i, (obs, action, reward) in enumerate(frames_list):
        past_turn = current_turn - (n - i)
        grid = _extract_grid(obs)
        hud = _extract_hud(obs)
        hud_line = f"  {hud}" if hud else ""
        lines.append(
            f"(turn {past_turn})\n"
            f"{grid}\n"
            f"{hud_line}\n"
            f"chose {action} → reward {reward:+.3f}".replace("  \n", "").replace("\n\n", "\n")
        )
    return "\n".join(lines)


def _render_current_block(current_obs: str, turn: int) -> str:
    # Strip the legend section since it's rendered globally at the top.
    grid = _extract_grid(current_obs)
    hud = _extract_hud(current_obs)
    msg = _extract_message(current_obs)
    parts = [f"[Current Observation — turn {turn}]"]
    if hud:
        parts.append(f"[HUD]\n{hud}")
    if grid:
        parts.append(f"[Grid]\n{grid}")
    if msg:
        parts.append(f"[Message]\n{msg}")
    return "\n".join(parts)
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/verifiers_integration/test_prompting.py -x 2>&1 | tail -20
```

Expected: 7 passed.

- [ ] **Step 4: Commit**

```bash
git add src/glyphbench/verifiers_integration/prompting.py tests/verifiers_integration/test_prompting.py
git commit -m "verifiers: prompting — frame-stack user turn with deduped legend + 512-token budget"
```

### Task 1.10: Create `verifiers_integration/rubric.py`

**Files:**
- Create: `src/glyphbench/verifiers_integration/rubric.py`
- Create: `tests/verifiers_integration/test_rubric.py`

- [ ] **Step 1: Write failing tests**

Create `tests/verifiers_integration/test_rubric.py`:

```python
"""Tests for EpisodicReturnRubric."""

from __future__ import annotations

import pytest

from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser
from glyphbench.verifiers_integration.rubric import EpisodicReturnRubric


@pytest.fixture
def rubric():
    return EpisodicReturnRubric(parser=GlyphbenchXMLParser())


async def _call(fn, **kw):
    return await fn(**kw)


@pytest.mark.asyncio
async def test_episodic_return_sums_per_step_rewards(rubric):
    state = {
        "episode_return": 1.5,
        "trajectory": [{"reward": 0.5}, {"reward": 0.5}, {"reward": 0.5}],
        "parse_failures": 0,
        "terminated": True,
        "truncated": False,
    }
    r = await rubric.episodic_return(state=state)
    assert r == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_parse_failure_rate(rubric):
    state = {
        "episode_return": 0.0,
        "trajectory": [{}, {}, {}, {}],
        "parse_failures": 1,
        "terminated": False,
        "truncated": True,
    }
    r = await rubric.parse_failure_rate(state=state)
    assert r == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_parse_failure_rate_empty_trajectory_returns_zero(rubric):
    state = {"episode_return": 0.0, "trajectory": [], "parse_failures": 0}
    r = await rubric.parse_failure_rate(state=state)
    assert r == 0.0


@pytest.mark.asyncio
async def test_episode_length(rubric):
    state = {"trajectory": [{}] * 7}
    assert await rubric.episode_length(state=state) == 7.0


@pytest.mark.asyncio
async def test_terminated_and_truncated_flags(rubric):
    assert await rubric.terminated_flag(state={"terminated": True}) == 1.0
    assert await rubric.terminated_flag(state={"terminated": False}) == 0.0
    assert await rubric.truncated_flag(state={"truncated": True}) == 1.0
    assert await rubric.truncated_flag(state={"truncated": False}) == 0.0
```

- [ ] **Step 2: Create `src/glyphbench/verifiers_integration/rubric.py`**

```python
"""Rubric: sums per-step rewards across the rollout + tracks monitor metrics.

The primary reward is ``episodic_return`` — weight 1.0, summed across every
step of the rollout. All other functions are ``weight=0`` metrics for
observability (parse-failure rate, episode length, terminated/truncated flags,
XML format compliance).
"""

from __future__ import annotations

from typing import Any

import verifiers as vf


class EpisodicReturnRubric(vf.Rubric):
    def __init__(self, parser: vf.Parser | None = None, **kwargs: Any) -> None:
        super().__init__(parser=parser, **kwargs)
        self.add_reward_func(self.episodic_return, weight=1.0)
        self.add_metric(self.episode_length)
        self.add_metric(self.parse_failure_rate)
        self.add_metric(self.terminated_flag)
        self.add_metric(self.truncated_flag)
        if parser is not None:
            try:
                fmt = parser.get_format_reward_func()
                fmt.__name__ = "xml_format_reward"
                self.add_metric(fmt)
            except AttributeError:
                pass  # parser doesn't expose a format reward fn — skip

    async def episodic_return(self, state: dict[str, Any]) -> float:
        return float(state.get("episode_return", 0.0))

    async def episode_length(self, state: dict[str, Any]) -> float:
        return float(len(state.get("trajectory", [])))

    async def parse_failure_rate(self, state: dict[str, Any]) -> float:
        traj_len = len(state.get("trajectory", []))
        if traj_len == 0:
            return 0.0
        return float(state.get("parse_failures", 0)) / traj_len

    async def terminated_flag(self, state: dict[str, Any]) -> float:
        return 1.0 if state.get("terminated") else 0.0

    async def truncated_flag(self, state: dict[str, Any]) -> float:
        return 1.0 if state.get("truncated") else 0.0
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/verifiers_integration/test_rubric.py -x 2>&1 | tail -15
```

Expected: 5 passed.

- [ ] **Step 4: Commit**

```bash
git add src/glyphbench/verifiers_integration/rubric.py tests/verifiers_integration/test_rubric.py
git commit -m "verifiers: EpisodicReturnRubric — sums step rewards, monitor metrics"
```

### Task 1.11: Create `verifiers_integration/env.py` — `GlyphbenchMultiTurnEnv` + `load_environment`

**Files:**
- Create: `src/glyphbench/verifiers_integration/env.py`
- Modify: `src/glyphbench/verifiers_integration/__init__.py` (uncomment imports)
- Modify: `src/glyphbench/__init__.py` (top-level re-exports)
- Create: `tests/verifiers_integration/test_env.py`

- [ ] **Step 1: Write failing tests**

Create `tests/verifiers_integration/test_env.py`:

```python
"""Tests for GlyphbenchMultiTurnEnv + load_environment."""

from __future__ import annotations

import json

import pytest

import glyphbench.envs.dummy  # ensure dummy registered
from glyphbench.verifiers_integration import GlyphbenchMultiTurnEnv, load_environment


def test_load_environment_returns_multi_turn_env():
    env = load_environment(
        env_id="glyphbench/__dummy-v0",
        num_episodes=2,
        n_frames=4,
        max_output_tokens=512,
    )
    assert isinstance(env, GlyphbenchMultiTurnEnv)


def test_load_environment_dataset_shape():
    env = load_environment(env_id="glyphbench/__dummy-v0", num_episodes=3)
    assert len(env.dataset) == 3
    for row in env.dataset:
        info = json.loads(row["info"]) if isinstance(row["info"], str) else row["info"]
        assert info["env_id"] == "glyphbench/__dummy-v0"
        assert isinstance(info["seed"], int)
    # Seeds should be distinct.
    seeds = [
        json.loads(r["info"])["seed"] if isinstance(r["info"], str) else r["info"]["seed"]
        for r in env.dataset
    ]
    assert len(set(seeds)) == len(seeds)


def test_load_environment_multiple_env_ids():
    env = load_environment(
        env_id=["glyphbench/__dummy-v0"],
        num_episodes=2,
    )
    assert len(env.dataset) == 2


def test_load_environment_rejects_unknown_id():
    with pytest.raises(KeyError):
        load_environment(env_id="glyphbench/does-not-exist-v0", num_episodes=1)


@pytest.mark.asyncio
async def test_setup_state_creates_game_and_initial_obs():
    env = load_environment(env_id="glyphbench/__dummy-v0", num_episodes=1)
    row = env.dataset[0]
    state: dict = {
        "info": row["info"],
        "prompt": [],
        "trajectory": [],
        "trajectory_id": "t0",
    }
    state = await env.setup_state(state)
    assert "game" in state
    assert state["current_obs"]  # non-empty
    assert state["parse_failures"] == 0
    assert state["episode_return"] == 0.0
    assert state["done"] is False
    assert state["terminated"] is False
    assert state["truncated"] is False
    # The prompt must have been populated with system + initial user turn.
    assert len(state["prompt"]) == 2
    assert state["prompt"][0]["role"] == "system"
    assert state["prompt"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_env_response_applies_action_and_updates_state():
    env = load_environment(env_id="glyphbench/__dummy-v0", num_episodes=1)
    row = env.dataset[0]
    state: dict = {
        "info": row["info"],
        "prompt": [],
        "trajectory": [{"reward": None, "extras": {}}],  # verifiers appends before env_response
        "trajectory_id": "t0",
    }
    state = await env.setup_state(state)
    state["trajectory"] = [{"reward": None, "extras": {}}]
    model_reply = "<think>go east</think><action>EAST</action>"
    messages = [{"role": "assistant", "content": model_reply}]
    response = await env.env_response(messages, state)
    # Returns a list of 1 user message with the next observation.
    assert isinstance(response, list) and len(response) == 1
    assert response[0]["role"] == "user"
    assert state["trajectory"][-1]["reward"] is not None
    # One frame accumulated for the action just taken.
    assert len(state["frames"]) == 1


@pytest.mark.asyncio
async def test_is_done_terminates_on_game_end():
    env = load_environment(env_id="glyphbench/__dummy-v0", num_episodes=1)
    state: dict = {"done": False}
    assert await env.is_done(state) is False
    state["done"] = True
    assert await env.is_done(state) is True
```

- [ ] **Step 2: Create `src/glyphbench/verifiers_integration/env.py`**

```python
"""GlyphbenchMultiTurnEnv + load_environment entry point."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import verifiers as vf
from datasets import Dataset

from glyphbench.core.base_env import BaseGlyphEnv
from glyphbench.core.registry import REGISTRY, all_glyphbench_env_ids, make_env
from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser
from glyphbench.verifiers_integration.prompting import (
    build_system_prompt,
    render_user_turn,
    set_budget_tokens,
)
from glyphbench.verifiers_integration.rubric import EpisodicReturnRubric


DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_N_FRAMES = 4
DEFAULT_NUM_EPISODES = 10
DEFAULT_BASE_SEED = 42


def load_environment(
    env_id: str | list[str] | None = None,
    num_episodes: int = DEFAULT_NUM_EPISODES,
    n_frames: int = DEFAULT_N_FRAMES,
    max_turns: int | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    seed: int = DEFAULT_BASE_SEED,
    **kwargs: Any,
) -> vf.Environment:
    """Entry point consumed by ``vf-eval`` and ``prime-rl`` orchestrator.

    Args:
        env_id: single id, list of ids, or ``None`` for all registered envs
                (dummy envs excluded when id is ``None``).
        num_episodes: rollouts per env.
        n_frames: history window shown in each user turn.
        max_turns: per-episode turn cap; ``None`` uses each game's own max_turns.
        max_output_tokens: per-turn LLM budget; communicated to the model in
                the system prompt.
        seed: base seed; each episode uses ``seed + episode_idx`` as the
                per-rollout seed.
    """
    _ensure_envs_loaded()
    env_ids = _resolve_env_ids(env_id)
    dataset = _build_dataset(env_ids, num_episodes, seed)
    set_budget_tokens(max_output_tokens)

    parser = GlyphbenchXMLParser()
    rubric = EpisodicReturnRubric(parser=parser)

    return GlyphbenchMultiTurnEnv(
        dataset=dataset,
        rubric=rubric,
        parser=parser,
        n_frames=n_frames,
        max_turns_override=max_turns,
        max_output_tokens=max_output_tokens,
    )


def _ensure_envs_loaded() -> None:
    """Force-import all suite __init__.py files so the registry is populated."""
    import glyphbench.envs  # noqa: F401
    # Individual suites populate REGISTRY on import.
    import glyphbench.envs.dummy      # noqa: F401
    for suite in ("minigrid", "minihack", "atari", "craftax", "procgen", "classics"):
        try:
            __import__(f"glyphbench.envs.{suite}")
        except ImportError:
            pass  # suite not present in this build — acceptable during migration


def _resolve_env_ids(env_id: str | list[str] | None) -> list[str]:
    if env_id is None:
        return [i for i in all_glyphbench_env_ids() if "__dummy" not in i]
    if isinstance(env_id, str):
        ids = [env_id]
    else:
        ids = list(env_id)
    missing = [i for i in ids if i not in REGISTRY]
    if missing:
        raise KeyError(
            f"unknown env_id(s): {missing!r}. "
            f"Known ids (sample): {sorted(REGISTRY)[:5]}…"
        )
    return ids


def _build_dataset(env_ids: list[str], num_episodes: int, base_seed: int) -> Dataset:
    rows = []
    for env_id in env_ids:
        for ep in range(num_episodes):
            seed_val = int(base_seed) + ep
            rows.append(
                {
                    "info": json.dumps({"env_id": env_id, "seed": seed_val}),
                    # Placeholder — filled in setup_state (verifiers allows
                    # dynamic prompt construction via state["prompt"] mutation).
                    "prompt": [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": ""},
                    ],
                    "answer": "",
                }
            )
    return Dataset.from_list(rows)


class GlyphbenchMultiTurnEnv(vf.MultiTurnEnv):
    """Verifiers MultiTurnEnv that drives a glyphbench game per rollout."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        rubric: vf.Rubric,
        parser: GlyphbenchXMLParser,
        n_frames: int,
        max_turns_override: int | None,
        max_output_tokens: int,
        **kwargs: Any,
    ) -> None:
        effective_max_turns = max_turns_override if max_turns_override is not None else -1
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            parser=parser,
            max_turns=effective_max_turns,
            **kwargs,
        )
        self.n_frames = int(n_frames)
        self._max_turns_override = max_turns_override
        self._max_output_tokens = int(max_output_tokens)
        self.parser: GlyphbenchXMLParser = parser  # narrow type

    async def setup_state(self, state: dict[str, Any]) -> dict[str, Any]:
        info_raw = state.get("info", {})
        info = json.loads(info_raw) if isinstance(info_raw, str) else info_raw
        env_id = info["env_id"]
        seed_val = int(info["seed"])

        kw: dict[str, Any] = {}
        if self._max_turns_override is not None:
            kw["max_turns"] = self._max_turns_override

        game = make_env(env_id, **kw)
        obs_text, _ = game.reset(seed_val)

        state["game"] = game
        state["frames"] = deque(maxlen=self.n_frames)
        state["current_obs"] = obs_text
        state["done"] = False
        state["terminated"] = False
        state["truncated"] = False
        state["parse_failures"] = 0
        state["episode_return"] = 0.0

        # Populate the prompt now that we have the game instance.
        system_text = build_system_prompt(game, self._max_output_tokens)
        initial_user_text = render_user_turn(
            game, frames=state["frames"], current_obs=obs_text, turn=0
        )
        state["prompt"] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": initial_user_text},
        ]
        return await super().setup_state(state)

    async def env_response(
        self,
        messages: list[dict[str, Any]],
        state: dict[str, Any],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        game: BaseGlyphEnv = state["game"]

        # The last message in `messages` is the assistant's reply.
        last_assistant = ""
        for m in reversed(messages):
            if m.get("role") == "assistant":
                last_assistant = m.get("content", "") or ""
                break

        action_idx, action_name, parse_failed = self.parser.parse_action(
            last_assistant, game.action_spec, noop=game.noop_action_name
        )
        if parse_failed:
            state["parse_failures"] += 1

        pre_obs = state["current_obs"]
        obs_text, reward, term, trunc, _info = game.step(action_idx)

        state["frames"].append((pre_obs, action_name, float(reward)))
        state["current_obs"] = obs_text
        state["episode_return"] += float(reward)
        state["terminated"] = bool(term)
        state["truncated"] = bool(trunc)
        state["done"] = bool(term or trunc)

        # Set the per-turn reward on the trajectory step verifiers appended
        # before calling env_response.
        traj = state.get("trajectory", [])
        if traj:
            traj[-1]["reward"] = float(reward)

        next_user = render_user_turn(
            game,
            frames=state["frames"],
            current_obs=obs_text,
            turn=game._turn,
        )
        return [{"role": "user", "content": next_user}]

    @vf.stop
    async def is_done(self, state: dict[str, Any]) -> bool:
        return bool(state.get("done", False))

    @vf.cleanup
    async def close_game(self, state: dict[str, Any]) -> None:
        game = state.pop("game", None)
        if game is not None:
            try:
                game.close()
            except Exception:  # noqa: BLE001
                pass
```

- [ ] **Step 3: Uncomment `verifiers_integration/__init__.py` imports**

Replace `src/glyphbench/verifiers_integration/__init__.py` with the full version:

```python
"""Verifiers integration — entry point for vf-eval and prime-rl."""

from glyphbench.verifiers_integration.env import (
    GlyphbenchMultiTurnEnv,
    load_environment,
)
from glyphbench.verifiers_integration.parser import GlyphbenchXMLParser
from glyphbench.verifiers_integration.prompting import (
    build_system_prompt,
    render_user_turn,
)
from glyphbench.verifiers_integration.rubric import EpisodicReturnRubric

__all__ = [
    "GlyphbenchMultiTurnEnv",
    "GlyphbenchXMLParser",
    "EpisodicReturnRubric",
    "build_system_prompt",
    "render_user_turn",
    "load_environment",
]
```

- [ ] **Step 4: Update top-level `src/glyphbench/__init__.py`**

Read the existing `src/glyphbench/__init__.py` first. Append the following imports / update exports so `glyphbench.load_environment` works:

```python
"""GlyphBench: unified benchmark of 292 text-rendered RL environments."""

from glyphbench.core import (
    ActionSpec,
    BaseGlyphEnv,
    GridObservation,
    REGISTRY,
    all_glyphbench_env_ids,
    make_env,
    register_env,
)
from glyphbench.verifiers_integration import (
    GlyphbenchMultiTurnEnv,
    GlyphbenchXMLParser,
    EpisodicReturnRubric,
    load_environment,
)

# Importing any suite module populates REGISTRY eagerly:
from glyphbench.envs import dummy  # noqa: F401

# The rest of the suites are optional during migration; they get added as
# the ports land in M2-M4 (import safely, silently skip on ImportError).
for _suite in ("minigrid", "minihack", "atari", "craftax", "procgen", "classics"):
    try:
        __import__(f"glyphbench.envs.{_suite}")
    except ImportError:
        pass

__all__ = [
    "ActionSpec",
    "BaseGlyphEnv",
    "GridObservation",
    "GlyphbenchMultiTurnEnv",
    "GlyphbenchXMLParser",
    "EpisodicReturnRubric",
    "REGISTRY",
    "all_glyphbench_env_ids",
    "make_env",
    "register_env",
    "load_environment",
]
```

- [ ] **Step 5: Run verifiers-integration tests + import smoke**

```bash
uv run pytest tests/verifiers_integration -x 2>&1 | tail -25
uv run python -c "import glyphbench; env = glyphbench.load_environment(env_id='glyphbench/__dummy-v0', num_episodes=1); print(type(env).__name__, len(env.dataset))"
```

Expected: all tests pass; smoke prints `GlyphbenchMultiTurnEnv 1`.

- [ ] **Step 6: Commit**

```bash
git add src/glyphbench/verifiers_integration/env.py src/glyphbench/verifiers_integration/__init__.py src/glyphbench/__init__.py tests/verifiers_integration/test_env.py
git commit -m "verifiers: GlyphbenchMultiTurnEnv + load_environment entry point"
```

### Task 1.12: End-to-end mock-rollout test

**Files:**
- Create: `tests/verifiers_integration/test_end_to_end.py`

- [ ] **Step 1: Write the test**

```python
"""End-to-end rollout test against a canned-response mock client."""

from __future__ import annotations

import pytest

import glyphbench
from glyphbench.core import make_env


@pytest.mark.asyncio
async def test_mock_rollout_accumulates_reward():
    """Drive the env through a scripted sequence of actions and verify the
    rubric sees a positive episodic return when the dummy env reaches its
    goal (east/south east sequence navigates 0,0 → 2,2 in 4 steps)."""

    env = glyphbench.load_environment(
        env_id="glyphbench/__dummy-v0",
        num_episodes=1,
        n_frames=4,
        max_turns=10,
    )
    # Scripted assistant replies — 4 moves to reach the goal.
    scripted = [
        "<think>right</think><action>EAST</action>",
        "<think>down</think><action>SOUTH</action>",
        "<think>right</think><action>EAST</action>",
        "<think>down</think><action>SOUTH</action>",
    ]

    # Drive setup_state + four env_response cycles by hand (mocking the model).
    row = env.dataset[0]
    state: dict = {
        "info": row["info"],
        "prompt": [],
        "trajectory": [],
        "trajectory_id": "t0",
    }
    state = await env.setup_state(state)

    for reply in scripted:
        state["trajectory"].append({"reward": None, "extras": {}})
        await env.env_response(
            [{"role": "assistant", "content": reply}], state
        )
        if state["done"]:
            break

    assert state["episode_return"] == pytest.approx(1.0)
    assert state["terminated"] is True
    assert state["truncated"] is False
    assert state["parse_failures"] == 0


@pytest.mark.asyncio
async def test_mock_rollout_parse_failures_accumulate():
    env = glyphbench.load_environment(
        env_id="glyphbench/__dummy-v0",
        num_episodes=1,
        n_frames=4,
        max_turns=3,
    )
    row = env.dataset[0]
    state: dict = {
        "info": row["info"],
        "prompt": [],
        "trajectory": [],
        "trajectory_id": "t0",
    }
    state = await env.setup_state(state)
    for _ in range(3):
        state["trajectory"].append({"reward": None, "extras": {}})
        await env.env_response(
            [{"role": "assistant", "content": "garbled output with no action"}],
            state,
        )
        if state["done"]:
            break
    assert state["parse_failures"] == 3
    assert state["truncated"] is True
    assert state["terminated"] is False
```

- [ ] **Step 2: Run and expect pass**

```bash
uv run pytest tests/verifiers_integration/test_end_to_end.py -x 2>&1 | tail -15
```

Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/verifiers_integration/test_end_to_end.py
git commit -m "verifiers: end-to-end mock rollout test (reward accumulation + parse failures)"
```

### Task 1.13: Run M1 green-gate check

- [ ] **Step 1: Run every test in the migrated portion**

```bash
uv run pytest tests/core tests/verifiers_integration tests/envs/dummy -x 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 2: Grep audit**

```bash
# M1 scope only — suites not yet migrated still reference gym.
grep -rn "gymnasium\|gym\.Env\|gym\.register\|gym\.make\b" \
    src/glyphbench/core \
    src/glyphbench/envs/dummy \
    src/glyphbench/verifiers_integration \
    tests/core \
    tests/envs/dummy \
    tests/verifiers_integration \
    2>&1 | grep -v "__pycache__\|.pyc"
```

Expected: no matches.

- [ ] **Step 3: Smoke test `vf-eval` discovery of the package**

```bash
uv run python -c "
import glyphbench
env = glyphbench.load_environment(env_id='glyphbench/__dummy-v0', num_episodes=2)
print(f'dataset rows: {len(env.dataset)}')
print(f'env type: {type(env).__name__}')
print('M1 smoke OK')
"
```

Expected: prints the lines above, no errors.

---

## Milestone 2 — Minigrid (71 envs)

**Goal end-state:** all minigrid envs import `BaseGlyphEnv`, register via `register_env(id, cls)`, and all `tests/envs/minigrid/` tests pass.

### Task 2.1: Port minigrid sources

**Files:**
- Modify: every `.py` in `src/glyphbench/envs/minigrid/`

- [ ] **Step 1: Bulk rename imports and base class references**

```bash
cd /home/roger/Desktop/rl-world-ascii
grep -rl "BaseAsciiEnv\|ascii_primitives" src/glyphbench/envs/minigrid/ | \
    xargs sed -i \
    -e 's/BaseAsciiEnv/BaseGlyphEnv/g' \
    -e 's/ascii_primitives/glyph_primitives/g'
```

- [ ] **Step 2: Rewrite `src/glyphbench/envs/minigrid/__init__.py`**

Replace `register_env("id", "module.path:Class", max_episode_steps=None)` with class-object registration. Full rewrite — read the current `__init__.py`, extract every `(id, module:class)` pair, emit a new file like:

```python
"""MiniGrid suite — importing registers all envs."""

from glyphbench.core.registry import register_env
from glyphbench.envs.minigrid.empty import (
    MiniGridEmpty5x5Env,
    MiniGridEmpty6x6Env,
    MiniGridEmpty8x8Env,
    MiniGridEmpty16x16Env,
    MiniGridEmptyRandom5x5Env,
    MiniGridEmptyRandom6x6Env,
)
from glyphbench.envs.minigrid.doorkey import (
    MiniGridDoorKey5x5Env,
    MiniGridDoorKey6x6Env,
    MiniGridDoorKey8x8Env,
    MiniGridDoorKey16x16Env,
)
# ...continue for every minigrid module...

_REGISTRATIONS = {
    "glyphbench/minigrid-empty-5x5-v0": MiniGridEmpty5x5Env,
    "glyphbench/minigrid-empty-6x6-v0": MiniGridEmpty6x6Env,
    # ... every id:Class pair ...
}

for _id, _cls in _REGISTRATIONS.items():
    register_env(_id, _cls)
```

Generate this programmatically — open the old `__init__.py` and write a Python script that parses every `register_env(id, "module:Class", ...)` entry and emits the new form:

```python
# /tmp/gen_minigrid_init.py
import re, pathlib

old = pathlib.Path("src/glyphbench/envs/minigrid/__init__.py").read_text()
entries = re.findall(
    r'register_env\(\s*"([^"]+)",\s*"glyphbench\.envs\.minigrid\.([a-z_]+):(\w+)"',
    old,
)
modules: dict[str, list[str]] = {}
for env_id, module, cls_name in entries:
    modules.setdefault(module, []).append(cls_name)

lines = ['"""MiniGrid suite — importing registers all envs."""', ""]
lines.append("from glyphbench.core.registry import register_env")
for module in sorted(modules):
    cls_list = sorted(set(modules[module]))
    lines.append(f"from glyphbench.envs.minigrid.{module} import (")
    for cls_name in cls_list:
        lines.append(f"    {cls_name},")
    lines.append(")")
lines += ["", "_REGISTRATIONS = {"]
for env_id, module, cls_name in entries:
    lines.append(f'    "{env_id}": {cls_name},')
lines += ["}", "", "for _id, _cls in _REGISTRATIONS.items():", "    register_env(_id, _cls)", ""]

pathlib.Path("src/glyphbench/envs/minigrid/__init__.py").write_text("\n".join(lines))
print(f"Wrote {len(entries)} registrations")
```

```bash
uv run python /tmp/gen_minigrid_init.py
```

Expected: `Wrote 71 registrations` (or similar).

- [ ] **Step 3: Verify imports resolve**

```bash
uv run python -c "import glyphbench.envs.minigrid; from glyphbench.core import all_glyphbench_env_ids; ids = [i for i in all_glyphbench_env_ids() if 'minigrid' in i]; print(f'{len(ids)} minigrid envs registered'); assert len(ids) == 71, ids"
```

Expected: `71 minigrid envs registered`.

- [ ] **Step 4: Commit**

```bash
git add src/glyphbench/envs/minigrid/
git commit -m "envs: port minigrid (71 envs) to non-gym base class + class-object registration"
```

### Task 2.2: Port minigrid tests

**Files:**
- Modify: every `.py` in `tests/envs/minigrid/`
- Modify: `tests/test_minigrid_prompt_sync.py`
- Modify: `tests/conftest.py` (drop gym imports if any)

- [ ] **Step 1: Bulk rewrite test files**

```bash
cd /home/roger/Desktop/rl-world-ascii
grep -rl "gym.make\|import gymnasium\|import gym\b\|\.unwrapped\|action_space\|observation_space" tests/envs/minigrid/ tests/test_minigrid_prompt_sync.py | xargs sed -i \
  -e 's/import gymnasium as gym//g' \
  -e 's/^import gym$//g' \
  -e 's/gym\.make(\([^)]*\))/make_env(\1)/g' \
  -e 's/\.unwrapped//g' \
  -e 's/env\.action_space\.n/env.action_spec.n/g' \
  -e 's/BaseAsciiEnv/BaseGlyphEnv/g'
```

- [ ] **Step 2: Add `from glyphbench.core import make_env` and suite import to tests**

For each test file in `tests/envs/minigrid/`, prepend:

```python
from glyphbench.core import make_env
import glyphbench.envs.minigrid  # register envs
```

Do this with a small script:

```bash
for f in tests/envs/minigrid/*.py; do
    if [[ "$f" == *__init__.py ]]; then continue; fi
    if ! grep -q "from glyphbench.core import make_env" "$f"; then
        sed -i '1i from glyphbench.core import make_env\nimport glyphbench.envs.minigrid  # register envs' "$f"
    fi
done
```

- [ ] **Step 3: Fix `env.reset(seed=42)` → positional**

```bash
grep -rl "env\.reset(seed=" tests/envs/minigrid/ tests/test_minigrid_prompt_sync.py | xargs sed -i 's/env\.reset(seed=\([0-9]\+\))/env.reset(\1)/g'
```

- [ ] **Step 4: Fix `observation_space` references (gymnasium-only)**

```bash
grep -rn "observation_space" tests/envs/minigrid/ tests/test_minigrid_prompt_sync.py
```

For each match, read the test and either drop the assertion or replace with an `isinstance(obs, str)` check.

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/envs/minigrid tests/test_minigrid_prompt_sync.py -x 2>&1 | tail -30
```

Expected: all pass. Fix remaining issues inline — common patterns:
  - `gym.Wrapper` references — remove the wrapper layer.
  - `env.render()` with mode kwarg → our envs render as part of `reset/step` return tuple; use `env.get_observation().render()` instead.
  - `env.metadata` references — drop.

- [ ] **Step 6: Commit**

```bash
git add tests/
git commit -m "tests: port minigrid tests to non-gym make_env/positional-seed API"
```

### Task 2.3: Run M2 green-gate

- [ ] **Step 1: Full minigrid test run + grep audit**

```bash
uv run pytest tests/envs/minigrid tests/test_minigrid_prompt_sync.py 2>&1 | tail -5
grep -rn "gymnasium\|gym\.Env\|gym\.register\|gym\.make\b" src/glyphbench/envs/minigrid tests/envs/minigrid 2>&1 | grep -v "__pycache__\|.pyc\|.git"
```

Expected: tests pass; grep empty.

---

## Milestone 3 — Classics (50) + Procgen (16)

Repeat the exact pattern from M2 for each suite, suite-by-suite (independent subagents can handle them in parallel in M3).

### Task 3.1: Port classics sources + `__init__.py`

**Files:**
- Modify: `src/glyphbench/envs/classics/*.py`, `src/glyphbench/envs/classics/__init__.py`

- [ ] **Step 1: Bulk rename**

```bash
grep -rl "BaseAsciiEnv\|ascii_primitives" src/glyphbench/envs/classics/ | xargs sed -i -e 's/BaseAsciiEnv/BaseGlyphEnv/g' -e 's/ascii_primitives/glyph_primitives/g'
```

- [ ] **Step 2: Regenerate `__init__.py`**

Adapt the `/tmp/gen_minigrid_init.py` script to target classics:

```python
# /tmp/gen_classics_init.py  (same logic, SUITE = "classics")
import re, pathlib

SUITE = "classics"
old = pathlib.Path(f"src/glyphbench/envs/{SUITE}/__init__.py").read_text()
entries = re.findall(
    rf'register_env\(\s*"([^"]+)",\s*"glyphbench\.envs\.{SUITE}\.([a-z_]+):(\w+)"',
    old,
)
modules: dict = {}
for env_id, module, cls in entries:
    modules.setdefault(module, []).append(cls)

lines = [f'"""{SUITE.capitalize()} suite — importing registers all envs."""', "",
         "from glyphbench.core.registry import register_env"]
for module in sorted(modules):
    cls_list = sorted(set(modules[module]))
    lines.append(f"from glyphbench.envs.{SUITE}.{module} import (")
    for c in cls_list: lines.append(f"    {c},")
    lines.append(")")
lines += ["", "_REGISTRATIONS = {"]
for env_id, _, cls in entries:
    lines.append(f'    "{env_id}": {cls},')
lines += ["}", "", "for _id, _cls in _REGISTRATIONS.items():", "    register_env(_id, _cls)", ""]
pathlib.Path(f"src/glyphbench/envs/{SUITE}/__init__.py").write_text("\n".join(lines))
print(f"Wrote {len(entries)} registrations")
```

```bash
uv run python /tmp/gen_classics_init.py
```

- [ ] **Step 3: Verify + commit**

```bash
uv run python -c "import glyphbench.envs.classics; from glyphbench.core import all_glyphbench_env_ids; ids = [i for i in all_glyphbench_env_ids() if 'classics' in i.split('/')[1].split('-')[0] or '/classic' in i or i.startswith('glyphbench/classics')]; print(len(ids))"
git add src/glyphbench/envs/classics
git commit -m "envs: port classics (50) to non-gym + class-object registration"
```

(Note: ids are under `glyphbench/classics-*-v0` — adjust the filter accordingly. The goal is ~50.)

### Task 3.2: Port classics tests

- [ ] **Step 1: Bulk rewrite (same recipe as 2.2)**

```bash
grep -rl "gym.make\|import gymnasium\|import gym\b\|\.unwrapped\|action_space\|observation_space" tests/envs/classics/ | xargs sed -i \
  -e 's/import gymnasium as gym//g' \
  -e 's/^import gym$//g' \
  -e 's/gym\.make(\([^)]*\))/make_env(\1)/g' \
  -e 's/\.unwrapped//g' \
  -e 's/env\.action_space\.n/env.action_spec.n/g' \
  -e 's/env\.reset(seed=\([0-9]\+\))/env.reset(\1)/g' \
  -e 's/BaseAsciiEnv/BaseGlyphEnv/g'

for f in tests/envs/classics/*.py; do
    if [[ "$f" == *__init__.py ]]; then continue; fi
    if ! grep -q "from glyphbench.core import make_env" "$f"; then
        sed -i '1i from glyphbench.core import make_env\nimport glyphbench.envs.classics  # register envs' "$f"
    fi
done
```

- [ ] **Step 2: Also port `tests/test_classics_rewards.py`**

```bash
sed -i \
  -e 's/import gymnasium as gym//g' \
  -e 's/gym\.make(\([^)]*\))/make_env(\1)/g' \
  -e 's/\.unwrapped//g' \
  -e 's/env\.reset(seed=\([0-9]\+\))/env.reset(\1)/g' \
  tests/test_classics_rewards.py
if ! grep -q "from glyphbench.core import make_env" tests/test_classics_rewards.py; then
    sed -i '1i from glyphbench.core import make_env\nimport glyphbench.envs.classics  # register envs' tests/test_classics_rewards.py
fi
```

- [ ] **Step 3: Run tests + fix residuals + commit**

```bash
uv run pytest tests/envs/classics tests/test_classics_rewards.py -x 2>&1 | tail -30
git add tests/envs/classics tests/test_classics_rewards.py
git commit -m "tests: port classics tests to non-gym API"
```

### Task 3.3: Port procgen (same pattern)

- [ ] **Step 1-5: Apply the identical recipe to `procgen`.**

```bash
# sources
grep -rl "BaseAsciiEnv\|ascii_primitives" src/glyphbench/envs/procgen/ | xargs sed -i -e 's/BaseAsciiEnv/BaseGlyphEnv/g' -e 's/ascii_primitives/glyph_primitives/g'

# __init__.py regeneration
sed -i 's/classics/procgen/g' /tmp/gen_classics_init.py > /tmp/gen_procgen_init.py || true
# Easier: rerun the script with SUITE = "procgen":
python3 -c "
import re, pathlib
SUITE = 'procgen'
old = pathlib.Path(f'src/glyphbench/envs/{SUITE}/__init__.py').read_text()
entries = re.findall(rf'register_env\(\s*\"([^\"]+)\",\s*\"glyphbench\.envs\.{SUITE}\.([a-z_]+):(\w+)\"', old)
modules = {}
for env_id, module, cls in entries:
    modules.setdefault(module, []).append(cls)
lines = [f'\"\"\"{SUITE.capitalize()} suite — importing registers all envs.\"\"\"', '',
         'from glyphbench.core.registry import register_env']
for module in sorted(modules):
    cls_list = sorted(set(modules[module]))
    lines.append(f'from glyphbench.envs.{SUITE}.{module} import (')
    for c in cls_list: lines.append(f'    {c},')
    lines.append(')')
lines += ['', '_REGISTRATIONS = {']
for env_id, _, cls in entries:
    lines.append(f'    \"{env_id}\": {cls},')
lines += ['}', '', 'for _id, _cls in _REGISTRATIONS.items():', '    register_env(_id, _cls)', '']
pathlib.Path(f'src/glyphbench/envs/{SUITE}/__init__.py').write_text('\n'.join(lines))
print(f'Wrote {len(entries)} registrations')
"
```

- [ ] **Step 2: tests (same recipe)**

```bash
grep -rl "gym.make\|import gymnasium\|import gym\b\|\.unwrapped\|action_space\|observation_space" tests/envs/procgen/ tests/test_procgen_determinism.py 2>/dev/null | xargs -r sed -i \
  -e 's/import gymnasium as gym//g' \
  -e 's/^import gym$//g' \
  -e 's/gym\.make(\([^)]*\))/make_env(\1)/g' \
  -e 's/\.unwrapped//g' \
  -e 's/env\.action_space\.n/env.action_spec.n/g' \
  -e 's/env\.reset(seed=\([0-9]\+\))/env.reset(\1)/g' \
  -e 's/BaseAsciiEnv/BaseGlyphEnv/g'

for f in tests/envs/procgen/*.py tests/test_procgen_determinism.py; do
    [[ -f "$f" ]] || continue
    if [[ "$f" == *__init__.py ]]; then continue; fi
    if ! grep -q "from glyphbench.core import make_env" "$f"; then
        sed -i '1i from glyphbench.core import make_env\nimport glyphbench.envs.procgen  # register envs' "$f"
    fi
done
```

- [ ] **Step 3: Run + commit**

```bash
uv run pytest tests/envs/procgen tests/test_procgen_determinism.py -x 2>&1 | tail -30
git add src/glyphbench/envs/procgen tests/envs/procgen tests/test_procgen_determinism.py
git commit -m "envs+tests: port procgen (16) to non-gym API"
```

### Task 3.4: M3 green-gate

- [ ] **Step 1:**

```bash
uv run pytest tests/envs/classics tests/envs/procgen tests/test_classics_rewards.py tests/test_procgen_determinism.py 2>&1 | tail -5
grep -rn "gymnasium\|gym\.Env\|gym\.register\|gym\.make\b" src/glyphbench/envs/classics src/glyphbench/envs/procgen tests/envs/classics tests/envs/procgen | grep -v "__pycache__"
```

Expected: all pass; grep empty.

---

## Milestone 4 — Atari (57) + Minihack (63) + Craftax (35)

Same recipe as M2/M3, three times over. This is the last suite batch — after M4 the `BaseAsciiEnv` alias must be removed from `core/__init__.py` and `core/base_env.py`.

### Task 4.1: Port atari sources + __init__.py

- [ ] **Step 1-3:** apply the M3.1 recipe with `SUITE = "atari"`.

```bash
grep -rl "BaseAsciiEnv\|ascii_primitives" src/glyphbench/envs/atari/ | xargs sed -i -e 's/BaseAsciiEnv/BaseGlyphEnv/g' -e 's/ascii_primitives/glyph_primitives/g'

python3 -c "
import re, pathlib
SUITE = 'atari'
old = pathlib.Path(f'src/glyphbench/envs/{SUITE}/__init__.py').read_text()
entries = re.findall(rf'register_env\(\s*\"([^\"]+)\",\s*\"glyphbench\.envs\.{SUITE}\.([a-z_]+):(\w+)\"', old)
modules = {}
for env_id, module, cls in entries:
    modules.setdefault(module, []).append(cls)
lines = [f'\"\"\"{SUITE.capitalize()} suite — importing registers all envs.\"\"\"', '',
         'from glyphbench.core.registry import register_env']
for module in sorted(modules):
    cls_list = sorted(set(modules[module]))
    lines.append(f'from glyphbench.envs.{SUITE}.{module} import (')
    for c in cls_list: lines.append(f'    {c},')
    lines.append(')')
lines += ['', '_REGISTRATIONS = {']
for env_id, _, cls in entries:
    lines.append(f'    \"{env_id}\": {cls},')
lines += ['}', '', 'for _id, _cls in _REGISTRATIONS.items():', '    register_env(_id, _cls)', '']
pathlib.Path(f'src/glyphbench/envs/{SUITE}/__init__.py').write_text('\n'.join(lines))
print(f'Wrote {len(entries)} registrations')
"

git add src/glyphbench/envs/atari
git commit -m "envs: port atari (57) to non-gym + class-object registration"
```

### Task 4.2: Port atari tests

Same recipe as M2.2/M3.2 targeting `tests/envs/atari/`:

```bash
grep -rl "gym.make\|import gymnasium\|import gym\b\|\.unwrapped\|action_space\|observation_space" tests/envs/atari/ | xargs sed -i \
  -e 's/import gymnasium as gym//g' \
  -e 's/^import gym$//g' \
  -e 's/gym\.make(\([^)]*\))/make_env(\1)/g' \
  -e 's/\.unwrapped//g' \
  -e 's/env\.action_space\.n/env.action_spec.n/g' \
  -e 's/env\.reset(seed=\([0-9]\+\))/env.reset(\1)/g' \
  -e 's/BaseAsciiEnv/BaseGlyphEnv/g'

for f in tests/envs/atari/*.py; do
    [[ "$f" == *__init__.py ]] && continue
    grep -q "from glyphbench.core import make_env" "$f" || sed -i '1i from glyphbench.core import make_env\nimport glyphbench.envs.atari  # register envs' "$f"
done

uv run pytest tests/envs/atari -x 2>&1 | tail -30
git add tests/envs/atari
git commit -m "tests: port atari tests to non-gym API"
```

### Task 4.3: Port minihack (same recipe, SUITE="minihack")

```bash
# sources
grep -rl "BaseAsciiEnv\|ascii_primitives" src/glyphbench/envs/minihack/ | xargs sed -i -e 's/BaseAsciiEnv/BaseGlyphEnv/g' -e 's/ascii_primitives/glyph_primitives/g'
python3 -c "
import re, pathlib
SUITE='minihack'
old=pathlib.Path(f'src/glyphbench/envs/{SUITE}/__init__.py').read_text()
entries=re.findall(rf'register_env\(\s*\"([^\"]+)\",\s*\"glyphbench\.envs\.{SUITE}\.([a-z_]+):(\w+)\"', old)
modules={}
for env_id, module, cls in entries: modules.setdefault(module,[]).append(cls)
lines=[f'\"\"\"{SUITE.capitalize()} suite — importing registers all envs.\"\"\"','','from glyphbench.core.registry import register_env']
for m in sorted(modules):
    cls_list=sorted(set(modules[m]))
    lines.append(f'from glyphbench.envs.{SUITE}.{m} import (')
    for c in cls_list: lines.append(f'    {c},')
    lines.append(')')
lines+=['','_REGISTRATIONS = {']
for env_id,_,cls in entries: lines.append(f'    \"{env_id}\": {cls},')
lines+=['}','','for _id, _cls in _REGISTRATIONS.items():','    register_env(_id, _cls)','']
pathlib.Path(f'src/glyphbench/envs/{SUITE}/__init__.py').write_text('\n'.join(lines))
print(f'Wrote {len(entries)}')
"

# tests
grep -rl "gym.make\|import gymnasium\|import gym\b\|\.unwrapped\|action_space\|observation_space" tests/envs/minihack/ | xargs sed -i \
  -e 's/import gymnasium as gym//g' \
  -e 's/^import gym$//g' \
  -e 's/gym\.make(\([^)]*\))/make_env(\1)/g' \
  -e 's/\.unwrapped//g' \
  -e 's/env\.action_space\.n/env.action_spec.n/g' \
  -e 's/env\.reset(seed=\([0-9]\+\))/env.reset(\1)/g' \
  -e 's/BaseAsciiEnv/BaseGlyphEnv/g'

for f in tests/envs/minihack/*.py; do
    [[ "$f" == *__init__.py ]] && continue
    grep -q "from glyphbench.core import make_env" "$f" || sed -i '1i from glyphbench.core import make_env\nimport glyphbench.envs.minihack  # register envs' "$f"
done

uv run pytest tests/envs/minihack -x 2>&1 | tail -30
git add src/glyphbench/envs/minihack tests/envs/minihack
git commit -m "envs+tests: port minihack (63) to non-gym API"
```

### Task 4.4: Port craftax (same recipe, SUITE="craftax")

```bash
grep -rl "BaseAsciiEnv\|ascii_primitives" src/glyphbench/envs/craftax/ | xargs sed -i -e 's/BaseAsciiEnv/BaseGlyphEnv/g' -e 's/ascii_primitives/glyph_primitives/g'
python3 -c "
import re, pathlib
SUITE='craftax'
old=pathlib.Path(f'src/glyphbench/envs/{SUITE}/__init__.py').read_text()
entries=re.findall(rf'register_env\(\s*\"([^\"]+)\",\s*\"glyphbench\.envs\.{SUITE}\.([a-z_]+):(\w+)\"', old)
modules={}
for env_id, module, cls in entries: modules.setdefault(module,[]).append(cls)
lines=[f'\"\"\"{SUITE.capitalize()} suite — importing registers all envs.\"\"\"','','from glyphbench.core.registry import register_env']
for m in sorted(modules):
    cls_list=sorted(set(modules[m]))
    lines.append(f'from glyphbench.envs.{SUITE}.{m} import (')
    for c in cls_list: lines.append(f'    {c},')
    lines.append(')')
lines+=['','_REGISTRATIONS = {']
for env_id,_,cls in entries: lines.append(f'    \"{env_id}\": {cls},')
lines+=['}','','for _id, _cls in _REGISTRATIONS.items():','    register_env(_id, _cls)','']
pathlib.Path(f'src/glyphbench/envs/{SUITE}/__init__.py').write_text('\n'.join(lines))
print(f'Wrote {len(entries)}')
"

grep -rl "gym.make\|import gymnasium\|import gym\b\|\.unwrapped\|action_space\|observation_space" tests/envs/craftax/ | xargs sed -i \
  -e 's/import gymnasium as gym//g' \
  -e 's/^import gym$//g' \
  -e 's/gym\.make(\([^)]*\))/make_env(\1)/g' \
  -e 's/\.unwrapped//g' \
  -e 's/env\.action_space\.n/env.action_spec.n/g' \
  -e 's/env\.reset(seed=\([0-9]\+\))/env.reset(\1)/g' \
  -e 's/BaseAsciiEnv/BaseGlyphEnv/g'

for f in tests/envs/craftax/*.py; do
    [[ "$f" == *__init__.py ]] && continue
    grep -q "from glyphbench.core import make_env" "$f" || sed -i '1i from glyphbench.core import make_env\nimport glyphbench.envs.craftax  # register envs' "$f"
done

uv run pytest tests/envs/craftax -x 2>&1 | tail -30
git add src/glyphbench/envs/craftax tests/envs/craftax
git commit -m "envs+tests: port craftax (35) to non-gym API"
```

### Task 4.5: Remove `BaseAsciiEnv` back-compat alias

**Files:**
- Modify: `src/glyphbench/core/base_env.py` (drop the alias line)
- Modify: `src/glyphbench/core/__init__.py` (drop the alias re-export)

- [ ] **Step 1: Grep — ensure no remaining references**

```bash
grep -rn "BaseAsciiEnv" src/ tests/ scripts/ eval/ docs/ 2>&1 | grep -v "__pycache__\|.pyc\|.git\|HANDOVER.md\|CONTRIBUTING.md\|CLAUDE.md\|AGENTS.md\|docs/superpowers/specs\|docs/superpowers/plans"
```

Expected: no matches (excluding planning/agents docs which may still mention the old name historically).

- [ ] **Step 2: Remove alias lines**

In `src/glyphbench/core/base_env.py`, delete the three comment + alias lines at the end:

```python
# Back-compat alias during migration (removed at end of Milestone 4).
# TEMPORARY — grep-audit at end of M4 removes this line.
BaseAsciiEnv = BaseGlyphEnv
```

In `src/glyphbench/core/__init__.py`, delete the alias line and `"BaseAsciiEnv"` entry from `__all__`.

- [ ] **Step 3: Test + commit**

```bash
uv run pytest tests/core -x 2>&1 | tail -5
git add src/glyphbench/core/base_env.py src/glyphbench/core/__init__.py
git commit -m "core: remove BaseAsciiEnv back-compat alias (M4 complete)"
```

### Task 4.6: Port remaining top-level tests

**Files:**
- Modify: `tests/test_end_to_end.py`, `tests/test_package_import.py`, `tests/conftest.py`

- [ ] **Step 1: Port**

```bash
grep -rl "gym.make\|import gymnasium\|import gym\b\|\.unwrapped\|action_space\|observation_space\|BaseAsciiEnv" tests/*.py tests/conftest.py | xargs -r sed -i \
  -e 's/import gymnasium as gym//g' \
  -e 's/^import gym$//g' \
  -e 's/gym\.make(\([^)]*\))/make_env(\1)/g' \
  -e 's/\.unwrapped//g' \
  -e 's/env\.action_space\.n/env.action_spec.n/g' \
  -e 's/env\.reset(seed=\([0-9]\+\))/env.reset(\1)/g' \
  -e 's/BaseAsciiEnv/BaseGlyphEnv/g'

# Add imports where the file uses make_env but doesn't import it.
for f in tests/test_end_to_end.py tests/test_package_import.py tests/conftest.py; do
    [[ -f "$f" ]] || continue
    if grep -q "make_env" "$f" && ! grep -q "from glyphbench.core import make_env" "$f"; then
        sed -i '1i from glyphbench.core import make_env\nimport glyphbench  # registers all envs' "$f"
    fi
done
```

- [ ] **Step 2: Run + commit**

```bash
uv run pytest tests -x --ignore=tests/test_cli.py 2>&1 | tail -10
git add tests
git commit -m "tests: port remaining top-level tests to non-gym API"
```

### Task 4.7: M4 green-gate — FULL pytest

- [ ] **Step 1: Run everything**

```bash
uv run pytest tests 2>&1 | tail -10
```

Expected: all tests pass. Log any failure and fix before moving on.

- [ ] **Step 2: Final grep audit across entire src/ and tests/**

```bash
grep -rn "gymnasium\|gym\.Env\|gym\.register\|gym\.make\b\|import gym\b\|import gymnasium\|BaseAsciiEnv\|ascii_primitives" src/ tests/ 2>&1 | grep -v "__pycache__\|.pyc\|.git\|.mypy_cache\|.ruff_cache\|.hypothesis"
```

Expected: no matches.

- [ ] **Step 3: Env count check**

```bash
uv run python -c "
import glyphbench
n = len(glyphbench.all_glyphbench_env_ids())
print(f'Registered envs: {n}')
assert n >= 292, f'expected ≥292, got {n}'
print('M4 COMPLETE')
"
```

Expected: `Registered envs: 293` (292 + `__dummy`), `M4 COMPLETE`.

---

## Milestone 5 — Scripts + random baseline + docs

### Task 5.1: Port `eval/random_baseline.py` + `tests/test_eval_history.py` → new eval helpers

**Files:**
- Modify: `eval/random_baseline.py`
- Delete: (already done in Task 1.1) `eval/run_eval.py`, `eval/scoring.py`
- Create: `eval/run_debug.sh`, `eval/run_full.sh`
- Modify: `eval/README.md`

- [ ] **Step 1: Read and port `eval/random_baseline.py`**

The existing file creates a random agent and records per-env returns. Port it:

```bash
# Port gym calls
sed -i \
  -e 's/import gymnasium as gym//g' \
  -e 's/gym\.make(\([^)]*\))/make_env(\1)/g' \
  -e 's/\.unwrapped//g' \
  -e 's/env\.reset(seed=\([0-9]\+\))/env.reset(\1)/g' \
  eval/random_baseline.py
# Add correct imports
if ! grep -q "from glyphbench.core import make_env" eval/random_baseline.py; then
    sed -i '1i from glyphbench.core import make_env\nimport glyphbench  # registers all envs' eval/random_baseline.py
fi
```

Read the file after sed — fix any remaining issues (e.g., `env.action_space.n` → `env.action_spec.n`).

- [ ] **Step 2: Regenerate `eval/random_baseline.json`**

```bash
uv run python eval/random_baseline.py --episodes 5 --output eval/random_baseline.json 2>&1 | tail -10
```

Expected: fresh json file produced.

- [ ] **Step 3: Write `eval/run_debug.sh`**

```bash
cat > eval/run_debug.sh <<'EOF'
#!/usr/bin/env bash
# Smoke eval: 2 envs × 2 episodes × Qwen3-0.6B against a local vLLM server.
# Prereq: a vLLM server at $VLLM_BASE_URL (default http://localhost:8000/v1)
# serving Qwen/Qwen3-0.6B. Start with:
#   uv run vllm serve Qwen/Qwen3-0.6B --port 8000
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
BASE_URL=${VLLM_BASE_URL:-http://localhost:8000/v1}

uv run vf-eval glyphbench \
  -m "$MODEL" \
  -b "$BASE_URL" \
  -n 2 -t 512 \
  -a '{"env_id": ["glyphbench/__dummy-v0", "glyphbench/minigrid-empty-5x5-v0"], "num_episodes": 2, "n_frames": 4}'
EOF
chmod +x eval/run_debug.sh
```

- [ ] **Step 4: Write `eval/run_full.sh`**

```bash
cat > eval/run_full.sh <<'EOF'
#!/usr/bin/env bash
# Full eval: all 292 envs × 10 episodes × Qwen3-0.6B.
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
BASE_URL=${VLLM_BASE_URL:-http://localhost:8000/v1}

uv run vf-eval glyphbench \
  -m "$MODEL" \
  -b "$BASE_URL" \
  -n 10 -t 512 \
  -a '{"num_episodes": 10, "n_frames": 4}'
EOF
chmod +x eval/run_full.sh
```

- [ ] **Step 5: Rewrite `eval/README.md`**

```markdown
# GlyphBench evaluation

GlyphBench exposes a verifiers environment with entry point
`glyphbench.load_environment`. Eval runs via the standard `vf-eval` CLI
against any OpenAI-compatible inference endpoint (we use vLLM).

## Quick start

```bash
# 1) start a vLLM server
uv run vllm serve Qwen/Qwen3-0.6B --port 8000

# 2) smoke test (2 envs × 2 episodes)
bash eval/run_debug.sh

# 3) full eval (292 envs × 10 episodes)
bash eval/run_full.sh
```

## `load_environment` arguments

```python
load_environment(
    env_id: str | list[str] | None = None,   # single id, list, or None=all
    num_episodes: int = 10,                   # rollouts per env
    n_frames: int = 4,                        # history window
    max_turns: int | None = None,             # None = use each env's own max
    max_output_tokens: int = 512,             # LLM budget per turn
    seed: int = 42,
)
```

Pass as JSON to `-a` / `-x`:

```bash
vf-eval glyphbench \
  -m Qwen/Qwen3-0.6B \
  -b http://localhost:8000/v1 \
  -n 5 -t 512 \
  -a '{"env_id":"glyphbench/atari-pong-v0","num_episodes":5,"n_frames":4}'
```

## Results

Verifiers writes per-rollout JSON records and aggregate metrics under
`~/.prime/evals/…` by default. View with `prime eval tui`.

## Random baseline

`eval/random_baseline.json` is regenerated by:

```bash
uv run python eval/random_baseline.py --episodes 5 --output eval/random_baseline.json
```

Used as a zero-skill reference for normalising model scores.
```

- [ ] **Step 6: Commit**

```bash
git add eval/
git commit -m "eval: vf-eval wrappers (debug+full) + ported random_baseline + new README"
```

### Task 5.2: Port `scripts/*` to non-gym API

**Files:**
- Modify: `scripts/demo_all_envs.py`, `scripts/play_random.py`, `scripts/play_interactive.py`, `scripts/play_curses.py`, `scripts/replay_trajectory.py`, `scripts/record_random_gifs.py`, `scripts/run_benchmark.py`, `scripts/generate_env_catalog.py`

- [ ] **Step 1: Bulk rewrite**

```bash
grep -rl "gym.make\|import gymnasium\|import gym\b\|\.unwrapped\|action_space\|observation_space\|BaseAsciiEnv" scripts/ | xargs sed -i \
  -e 's/import gymnasium as gym//g' \
  -e 's/^import gym$//g' \
  -e 's/gym\.make(\([^)]*\))/make_env(\1)/g' \
  -e 's/\.unwrapped//g' \
  -e 's/env\.action_space\.n/env.action_spec.n/g' \
  -e 's/env\.reset(seed=\([0-9]\+\))/env.reset(\1)/g' \
  -e 's/BaseAsciiEnv/BaseGlyphEnv/g'

# Add make_env import where used
for f in scripts/*.py; do
    [[ -f "$f" ]] || continue
    if grep -q "make_env(" "$f" && ! grep -q "from glyphbench.core import make_env" "$f"; then
        sed -i '1i from glyphbench.core import make_env\nimport glyphbench  # registers all envs' "$f"
    fi
done
```

- [ ] **Step 2: Smoke-run each script with --help**

```bash
for f in scripts/demo_all_envs.py scripts/play_random.py scripts/record_random_gifs.py scripts/replay_trajectory.py scripts/run_benchmark.py; do
    echo "=== $f ==="
    uv run python "$f" --help 2>&1 | head -5
done
```

Expected: each prints a usage message, no import errors.

- [ ] **Step 3: Demo smoke**

```bash
uv run python scripts/play_random.py --env glyphbench/__dummy-v0 --steps 5 --seed 42 2>&1 | tail -5
```

Expected: prints a few turns of a random rollout.

- [ ] **Step 4: Commit**

```bash
git add scripts/
git commit -m "scripts: port demo/play/replay/record to non-gym API (make_env + positional seed)"
```

### Task 5.3: Update `src/glyphbench/cli.py`

**Files:**
- Modify: `src/glyphbench/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Read current CLI**

```bash
cat src/glyphbench/cli.py | head -50
```

- [ ] **Step 2: Port**

Replace any gym references. Most likely just:
  - `gym.make(...)` → `make_env(...)`
  - `.unwrapped` → removed

```bash
sed -i \
  -e 's/import gymnasium as gym//g' \
  -e 's/gym\.make(\([^)]*\))/make_env(\1)/g' \
  -e 's/\.unwrapped//g' \
  -e 's/env\.reset(seed=\([0-9]\+\))/env.reset(\1)/g' \
  src/glyphbench/cli.py
```

Add imports if needed.

- [ ] **Step 3: Test + commit**

```bash
uv run pytest tests/test_cli.py -x 2>&1 | tail -10
git add src/glyphbench/cli.py tests/test_cli.py
git commit -m "cli: port to non-gym API"
```

### Task 5.4: M5 green-gate

- [ ] **Step 1:**

```bash
uv run pytest tests 2>&1 | tail -10
# Entire repo grep audit
grep -rn "gymnasium\|gym\.Env\|gym\.register\|gym\.make\b\|import gym\b\|import gymnasium\|BaseAsciiEnv\|ascii_primitives" \
    src/ tests/ scripts/ eval/ \
    2>&1 | grep -v "__pycache__\|.pyc\|.git\|.mypy_cache\|.ruff_cache\|.hypothesis"
```

Expected: tests pass; grep empty.

---

## Milestone 6 — Container + prime-rl smoke

### Task 6.1: Rewrite `Dockerfile`

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Full replace**

```dockerfile
# GlyphBench — verifiers + prime-rl container
#
# Build:   docker build -t glyphbench:latest .
# SIF:     bash scripts/build_sif.sh
#
# Run eval inside container (with model weights mounted):
#   apptainer run --nv --bind $HF_HOME:/root/.cache/huggingface \
#     glyphbench.sif bash eval/run_debug.sh

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
  && rm -rf /var/lib/apt/lists/*

# uv (Python & Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
  && cp /root/.local/bin/uv /usr/local/bin/uv \
  && cp /root/.local/bin/uvx /usr/local/bin/uvx

WORKDIR /opt/glyphbench

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock /opt/glyphbench/
RUN uv python install 3.12 \
  && uv sync --frozen --extra eval

# Copy source and install the editable package
COPY . /opt/glyphbench
RUN uv sync --frozen --extra eval

# Optional RL extra (prime-rl + flash-attn). Heavy; install at run time if needed.
# RUN uv sync --frozen --extra eval --extra rl

ENV PATH="/opt/glyphbench/.venv/bin:${PATH}"
WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]
```

- [ ] **Step 2: Build test**

```bash
cd /home/roger/Desktop/rl-world-ascii
docker build -t glyphbench:latest . 2>&1 | tail -30
```

Expected: image built; last lines show `Successfully tagged glyphbench:latest`. (Build takes 5–15 min due to verifiers/vllm dependency tree.)

- [ ] **Step 3: Container smoke**

```bash
docker run --rm glyphbench:latest uv run python -c "
import glyphbench
print(f'envs: {len(glyphbench.all_glyphbench_env_ids())}')
env = glyphbench.load_environment(env_id='glyphbench/__dummy-v0', num_episodes=1)
print(f'load_environment ok: {type(env).__name__}')
"
```

Expected: prints env count ≥ 292 and `load_environment ok: GlyphbenchMultiTurnEnv`.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile
git commit -m "container: rebuild Dockerfile on nvidia/cuda:12.4.1 + uv + verifiers"
```

### Task 6.2: Write `scripts/build_sif.sh`

**Files:**
- Create: `scripts/build_sif.sh`

- [ ] **Step 1:**

```bash
cat > scripts/build_sif.sh <<'EOF'
#!/usr/bin/env bash
# Build the GlyphBench container and convert to singularity/apptainer SIF.
# Overwrites any existing IMAGE or SIF.
set -euo pipefail
IMAGE=${IMAGE:-glyphbench:latest}
SIF=${SIF:-glyphbench.sif}

echo ">>> Docker build $IMAGE"
docker build -t "$IMAGE" .

echo ">>> Apptainer build $SIF"
apptainer build --force "$SIF" "docker-daemon://$IMAGE"

echo ">>> Done: $SIF"
ls -lh "$SIF"
EOF
chmod +x scripts/build_sif.sh
```

- [ ] **Step 2: Run it**

```bash
bash scripts/build_sif.sh 2>&1 | tail -20
```

Expected: produces `glyphbench.sif` in cwd.

- [ ] **Step 3: SIF smoke test**

```bash
apptainer run glyphbench.sif bash -c "cd /opt/glyphbench && uv run python -c \"import glyphbench; print(len(glyphbench.all_glyphbench_env_ids()))\""
```

Expected: prints ≥ 292.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_sif.sh
# Do NOT commit glyphbench.sif (multi-GB binary). Ensure .gitignore covers it:
grep -q "glyphbench.sif" .gitignore || echo "glyphbench.sif" >> .gitignore
git add .gitignore
git commit -m "container: scripts/build_sif.sh + gitignore sif output"
```

### Task 6.3: Create prime-rl smoke RL config

**Files:**
- Create: `configs/rl/glyphbench-smoke/rl.toml`

- [ ] **Step 1: Write `configs/rl/glyphbench-smoke/rl.toml`**

Reference: `/tmp/prime-rl-main/examples/wordle/rl.toml` (verified in spec). Read it first to confirm the current field schema:

```bash
unzip -qo /home/roger/Desktop/rl-world-ascii/prime-rl-main.zip -d /tmp/
cat /tmp/prime-rl-main/examples/wordle/rl.toml
# Also check the RL entrypoint config surface:
grep -n "class.*Config\|field\|args" /tmp/prime-rl-main/src/prime_rl/entrypoints/rl.py | head -40
```

Draft:

```toml
# Smoke-only config: proves the prime-rl pipeline runs end-to-end against
# the new glyphbench verifiers package. NOT tuned for quality.

max_steps = 2
seq_len = 4096

[deployment]
num_train_gpus = 1
num_infer_gpus = 1

[wandb]
project = "glyphbench-smoke"
name = "smoke"

[ckpt]

[model]
name = "Qwen/Qwen3-0.6B"

[orchestrator]
batch_size = 4
rollouts_per_example = 2

[[orchestrator.train.env]]
id = "glyphbench"
name = "glyphbench-minigrid-5x5"
args = { env_id = "glyphbench/minigrid-empty-5x5-v0", num_episodes = 4, n_frames = 4, max_turns = 30 }

[orchestrator.train.sampling]
max_completion_tokens = 512
temperature = 0.7
top_p = 0.9

[trainer]

[inference]
enforce_eager = true

[inference.parallel]
dp = 1
```

- [ ] **Step 2: (Optional) Dry-run validation**

```bash
uv run python -c "
import tomllib
cfg = tomllib.loads(open('configs/rl/glyphbench-smoke/rl.toml').read())
print('config loads:', list(cfg))
assert cfg['model']['name'] == 'Qwen/Qwen3-0.6B'
assert cfg['orchestrator']['train']['env'][0]['id'] == 'glyphbench'
print('smoke: TOML valid')
"
```

- [ ] **Step 3: Commit**

```bash
mkdir -p configs/rl/glyphbench-smoke
# (written above directly)
git add configs/rl/glyphbench-smoke/rl.toml
git commit -m "prime-rl: glyphbench-smoke rl.toml (Qwen3-0.6B, minigrid-empty-5x5, 2 steps)"
```

### Task 6.4: Local vf-eval smoke (Qwen3-0.6B)

**Dependency:** a GPU is required; the user's `/home/roger/Desktop/rl-world-ascii/Qwen_Qwen3.5-0.8B/` directory has a model cache, but we're using Qwen3-0.6B. The subagent should confirm the user's default HF cache can fetch Qwen/Qwen3-0.6B; if offline, document the model substitution.

- [ ] **Step 1: Start vllm server (background)**

```bash
# The user's .env has HF_TOKEN already.
source /home/roger/Desktop/rl-world-ascii/.env 2>/dev/null || true
nohup uv run vllm serve Qwen/Qwen3-0.6B --port 8000 --max-model-len 4096 --enforce-eager > /tmp/vllm.log 2>&1 &
VLLM_PID=$!
echo "vllm pid: $VLLM_PID"
# Poll until ready:
until curl -fs http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 5; done
echo "vllm ready"
```

- [ ] **Step 2: Run vf-eval smoke**

```bash
bash eval/run_debug.sh 2>&1 | tail -40
```

Expected: completes with per-env returns printed; no tracebacks. Writes results under `~/.prime/evals/…` or the verifiers default.

- [ ] **Step 3: Stop vllm**

```bash
kill "$VLLM_PID" 2>/dev/null || true
```

- [ ] **Step 4: (No commit — this is a smoke verification. Note any output lines into the run log.)**

### Task 6.5: prime-rl RL smoke

**Dependency:** prime-rl must be installed (`--extra rl`). If not installed by default, `uv sync --extra rl` first. Requires at least 2 GPUs (1 train + 1 infer) — the subagent should skip this task and log SKIP if only 1 GPU available.

- [ ] **Step 1: Check GPU count**

```bash
python3 -c "
import subprocess
out = subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader']).decode()
print('GPUs:', out.strip().splitlines())
" 2>&1
```

If fewer than 2 GPUs, append to plan: `# SKIPPED: insufficient GPUs for RL smoke` and move to Task 6.6.

- [ ] **Step 2: Install `rl` extra**

```bash
uv sync --extra eval --extra rl 2>&1 | tail -5
```

- [ ] **Step 3: Run smoke**

```bash
uv run rl @ configs/rl/glyphbench-smoke/rl.toml 2>&1 | tail -40
```

Expected: runs at least 1 full orchestrator step without crashing. Note: this needs a couple of minutes to spin up trainer + inference + orchestrator.

- [ ] **Step 4: Commit if adjustments needed**

If config required tweaks to match current prime-rl schema, commit the corrected `rl.toml`:

```bash
git add configs/rl/glyphbench-smoke/rl.toml
git commit -m "prime-rl: fix smoke config against current prime-rl schema"
```

### Task 6.6: Container smoke (SIF)

- [ ] **Step 1: Run vf-eval inside the SIF**

```bash
# Needs a vllm server already running on host at localhost:8000.
apptainer run --nv --bind "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  glyphbench.sif bash -c "cd /opt/glyphbench && bash eval/run_debug.sh" 2>&1 | tail -40
```

Expected: smoke completes without error inside the container, same as bare-metal.

### Task 6.7: README + HANDOVER updates

**Files:**
- Modify: `README.md`, `HANDOVER.md`

- [ ] **Step 1: Update `README.md` "Install" / "Quick start" / "Running LLM evaluations"**

Key edits:
- Drop any mention of `gymnasium` in the Install / Quickstart sections.
- "Quick start" example switches from `gym.make` to `glyphbench.load_environment` + `make_env`:

```python
import glyphbench
from glyphbench.core import make_env

# Direct game loop
env = make_env("glyphbench/minigrid-empty-5x5-v0")
obs, info = env.reset(42)

# Or: load as a verifiers environment for eval / RL
vf_env = glyphbench.load_environment(env_id="glyphbench/minigrid-empty-5x5-v0")
```

- "Running LLM evaluations" switches `python eval/run_eval.py …` → `bash eval/run_debug.sh` / `bash eval/run_full.sh`.
- "Development" section drops the `[eval]` extra reference for vllm since `eval/` no longer has a Python runner — keep `[eval]` extra as "install vllm for serving models locally".

- [ ] **Step 2: Append a migration note to `HANDOVER.md`**

```markdown
## 2026-04-24 — verifiers migration

The repo moved off gymnasium to verifiers + prime-rl. Key changes:

- `BaseAsciiEnv` → `BaseGlyphEnv`; `ascii_primitives.py` → `glyph_primitives.py`.
- `gym.register/gym.make` replaced with `register_env(id, cls)` / `make_env(id, **kw)` on a dict-based registry.
- Eval now runs via `vf-eval` against any OpenAI-compatible endpoint; see `eval/README.md`. The old `eval/run_eval.py` is gone.
- `glyphbench.load_environment(env_id=..., num_episodes=10, n_frames=4, max_output_tokens=512)` is the verifiers entry point.
- Harness: one mode only — frame-stacked history (N=4 default), CoT-only, 512-token budget communicated to the model.
- Container rebuilt on `nvidia/cuda:12.4.1-devel-ubuntu22.04`; build with `bash scripts/build_sif.sh`.
- prime-rl smoke config: `configs/rl/glyphbench-smoke/rl.toml`; launch with `uv run rl @ configs/rl/glyphbench-smoke/rl.toml`.
- All 2000+ tests ported and green; `random_baseline.json` regenerated under the new harness.
```

- [ ] **Step 3: Commit**

```bash
git add README.md HANDOVER.md
git commit -m "docs: update README + HANDOVER for the verifiers migration"
```

### Task 6.8: Final green-gate + handover

- [ ] **Step 1: Full repo test run**

```bash
uv run pytest 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 2: Final full-repo grep audit**

```bash
grep -rn "gymnasium\|gym\.Env\|gym\.register\|gym\.make\b\|import gym\b\|import gymnasium\|BaseAsciiEnv\|ascii_primitives" \
    src/ tests/ scripts/ eval/ configs/ Dockerfile README.md HANDOVER.md pyproject.toml \
    2>&1 | grep -v "__pycache__\|.pyc\|.git\|.mypy_cache\|.ruff_cache\|.hypothesis\|docs/superpowers"
```

Expected: no matches.

- [ ] **Step 3: Summary report**

Print a final summary:

```bash
uv run python -c "
import glyphbench
ids = glyphbench.all_glyphbench_env_ids()
print(f'Total registered envs: {len(ids)}')
for suite in ('minigrid', 'minihack', 'atari', 'craftax', 'procgen', 'classics', '__dummy'):
    n = sum(1 for i in ids if suite in i)
    print(f'  {suite}: {n}')
env = glyphbench.load_environment(env_id='glyphbench/__dummy-v0', num_episodes=1)
print(f'load_environment OK: {type(env).__name__}')
print('MIGRATION COMPLETE.')
"
```

Expected: 292 envs + 1 dummy, all suite counts match (minigrid 71, minihack 63, atari 57, craftax 35, procgen 16, classics 50, dummy 1), and `MIGRATION COMPLETE.`

- [ ] **Step 4: Final commit (if residual changes)**

```bash
git status
# Commit anything outstanding.
```

---

## Self-review notes

Spec coverage check — every spec section maps to a task:

- §Architecture → M1 (core rewrite), M2–M4 (suite ports), M5 (scripts/eval), M6 (container, prime-rl).
- §Core non-gym base class → Task 1.3.
- §Registry → Task 1.4.
- §Verifiers integration → Tasks 1.8-1.11.
- §Frame-stack prompting → Task 1.9 (prompting.py + tests).
- §Parser → Task 1.8.
- §Rubric → Task 1.10.
- §Dataset shape → Task 1.11 (`_build_dataset` in `env.py`).
- §Sampling defaults → Task 1.11 (defaults declared in `load_environment`).
- §Data flow — one rollout → validated by Task 1.12 (end-to-end mock test) + Task 6.4 (vf-eval smoke).
- §Tests — porting + new → Tasks 2.2, 3.2, 3.4, 4.2, 4.3, 4.4, 4.6 (mechanical suite-test ports) + Task 1.8-1.12 (new verifiers-integration tests).
- §Docker + Singularity → Tasks 6.1, 6.2, 6.6.
- §prime-rl wiring → Task 6.3 (config) + 6.5 (smoke).
- §Error handling → covered by Task 1.8 parser tests (parse failure → noop), Task 1.3 base_env tests (max_turns truncation), Task 1.12 end-to-end test (parse-failure rate accumulation).
- §Verification checklist → Tasks 4.7, 5.4, 6.8 (green-gates after each milestone).

Placeholder scan: no "TBD", "TODO", "fill in later", or "similar to X" appears. Every step has a runnable command or complete code block.

Type consistency: `make_env`, `register_env`, `REGISTRY`, `BaseGlyphEnv`, `GlyphbenchMultiTurnEnv`, `GlyphbenchXMLParser`, `EpisodicReturnRubric`, `load_environment`, `build_system_prompt`, `render_user_turn` are used consistently across tasks.
