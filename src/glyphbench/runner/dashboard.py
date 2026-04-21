"""Rich live dashboard showing per-env progress + cost."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from rich.console import Console
from rich.live import Live
from rich.table import Table


class DashboardProtocol(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def update_env(
        self,
        env_id: str,
        *,
        episodes_done: int,
        episodes_total: int,
        mean_return: float,
        mean_len: float,
    ) -> None: ...
    def update_totals(
        self,
        *,
        cost_used: float,
        budget: float | None,
        parse_failures: int,
        provider_errors: int,
        fallback_noops: int,
    ) -> None: ...
    def log_event(self, message: str) -> None: ...


class NullDashboard:
    """No-op dashboard used in tests and when config.dashboard is False."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def update_env(
        self,
        env_id: str,
        *,
        episodes_done: int,
        episodes_total: int,
        mean_return: float,
        mean_len: float,
    ) -> None: ...
    def update_totals(
        self,
        *,
        cost_used: float,
        budget: float | None,
        parse_failures: int,
        provider_errors: int,
        fallback_noops: int,
    ) -> None: ...
    def log_event(self, message: str) -> None: ...


@dataclass
class _EnvRow:
    episodes_done: int = 0
    episodes_total: int = 0
    mean_return: float = 0.0
    mean_len: float = 0.0


@dataclass
class _Totals:
    cost_used: float = 0.0
    budget: float | None = None
    parse_failures: int = 0
    provider_errors: int = 0
    fallback_noops: int = 0


class Dashboard:
    def __init__(
        self,
        *,
        run_id: str,
        model_id: str,
        provider: str,
        total_episodes: int,
    ) -> None:
        self.run_id = run_id
        self.model_id = model_id
        self.provider = provider
        self.total_episodes = total_episodes
        self._env_rows: dict[str, _EnvRow] = {}
        self._totals = _Totals()
        self._last_event: str = ""
        self._start_time: float = 0.0
        self._live: Live | None = None
        self._console = Console()

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._live = Live(
            self._render(), refresh_per_second=4, console=self._console, transient=False
        )
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.update(self._render())
            self._live.stop()
            self._live = None

    def update_env(
        self,
        env_id: str,
        *,
        episodes_done: int,
        episodes_total: int,
        mean_return: float,
        mean_len: float,
    ) -> None:
        row = self._env_rows.setdefault(env_id, _EnvRow())
        row.episodes_done = episodes_done
        row.episodes_total = episodes_total
        row.mean_return = mean_return
        row.mean_len = mean_len
        self._refresh()

    def update_totals(
        self,
        *,
        cost_used: float,
        budget: float | None,
        parse_failures: int,
        provider_errors: int,
        fallback_noops: int,
    ) -> None:
        self._totals = _Totals(
            cost_used=cost_used,
            budget=budget,
            parse_failures=parse_failures,
            provider_errors=provider_errors,
            fallback_noops=fallback_noops,
        )
        self._refresh()

    def log_event(self, message: str) -> None:
        self._last_event = message
        self._refresh()

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "env_rows": {k: vars(v) for k, v in self._env_rows.items()},
            "totals": vars(self._totals),
            "last_event": self._last_event,
        }

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> Any:
        table = Table(title=f"GlyphBench - {self.run_id}")
        table.add_column("Env")
        table.add_column("Episodes", justify="right")
        table.add_column("Mean Ret", justify="right")
        table.add_column("Mean Len", justify="right")
        for env_id, row in self._env_rows.items():
            table.add_row(
                env_id,
                f"{row.episodes_done}/{row.episodes_total}",
                f"{row.mean_return:.2f}",
                f"{row.mean_len:.1f}",
            )
        if self._totals.budget is not None:
            cost_line = f"Cost: ${self._totals.cost_used:.2f} / ${self._totals.budget:.2f}"
        else:
            cost_line = f"Cost: ${self._totals.cost_used:.2f} / unlimited"
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0.0
        table.caption = (
            f"Model: {self.model_id} ({self.provider})   "
            f"Elapsed: {elapsed:.0f}s   "
            f"{cost_line}   "
            f"Parse fails: {self._totals.parse_failures}   "
            f"Provider errs: {self._totals.provider_errors}   "
            f"Noop fallbacks: {self._totals.fallback_noops}"
            + (f"\n{self._last_event}" if self._last_event else "")
        )
        return table
