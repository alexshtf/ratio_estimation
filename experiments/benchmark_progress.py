"""Live progress helpers for the benchmark workflow."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rich.console import RenderableType

BENCHMARK_FLOAT_DISPLAY_WIDTH = 10


@dataclass(frozen=True, slots=True)
class _ProgressTableWidths:
    """Pinned column widths for the live rich benchmark progress table."""

    model: int
    split: int
    progress: int
    loss: int
    tune_sec_per_trial: int
    status: int


@dataclass(slots=True)
class _ProgressRowState:
    """Display state for one model row in the live benchmark progress table."""

    model: str
    split: str = "pending"
    progress: str = "--"
    last_loss: str = "--"
    best_loss: str = "--"
    tune_sec_per_trial: str = "--"
    status: str = "waiting"


class _BenchmarkProgress:
    """Common interface for live and no-op benchmark progress controllers."""

    enabled = False

    def __enter__(self) -> _BenchmarkProgress:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _ = exc_type, exc, traceback

    def update_row(self, model_name: str, **changes: str) -> None:
        """Apply display updates to one progress row."""
        _ = model_name, changes


class _NullBenchmarkProgress(_BenchmarkProgress):
    """Do nothing when the benchmark runs outside an interactive terminal."""


class _RichBenchmarkProgress(_BenchmarkProgress):
    """Render one live progress table for the full benchmark run."""

    enabled = True

    def __init__(self, model_names: list[str], widths: _ProgressTableWidths) -> None:
        from rich.console import Console
        from rich.live import Live

        self.model_order = model_names
        self.widths = widths
        self.rows = {model_name: _ProgressRowState(model=model_name) for model_name in model_names}
        self.console = Console()
        self.live = Live(
            self._render_table(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )

    def __enter__(self) -> _RichBenchmarkProgress:
        self.live.__enter__()
        self.live.update(self._render_table(), refresh=True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.live.update(self._render_table(), refresh=True)
        self.live.__exit__(exc_type, exc, traceback)

    def update_row(self, model_name: str, **changes: str) -> None:
        """Apply display updates to one live table row."""
        row = self.rows[model_name]
        for field_name, value in changes.items():
            setattr(row, field_name, value)
        self.live.update(self._render_table(), refresh=True)

    def _render_table(self) -> RenderableType:
        """Render the current table state as a rich table object."""
        from rich.table import Table

        table = Table(title="Benchmark Progress")
        table.add_column(
            "Model",
            width=self.widths.model,
            min_width=self.widths.model,
            no_wrap=True,
        )
        table.add_column(
            "Split",
            width=self.widths.split,
            min_width=self.widths.split,
            no_wrap=True,
        )
        table.add_column(
            "Progress",
            width=self.widths.progress,
            min_width=self.widths.progress,
            no_wrap=True,
        )
        table.add_column(
            "Last Loss",
            justify="right",
            width=self.widths.loss,
            min_width=self.widths.loss,
            no_wrap=True,
        )
        table.add_column(
            "Best Loss",
            justify="right",
            width=self.widths.loss,
            min_width=self.widths.loss,
            no_wrap=True,
        )
        table.add_column(
            "Tune Sec/Trial",
            justify="right",
            width=self.widths.tune_sec_per_trial,
            min_width=self.widths.tune_sec_per_trial,
            no_wrap=True,
        )
        table.add_column(
            "Status",
            width=self.widths.status,
            min_width=self.widths.status,
            no_wrap=True,
        )
        for model_name in self.model_order:
            row = self.rows[model_name]
            table.add_row(
                row.model,
                row.split,
                row.progress,
                row.last_loss,
                row.best_loss,
                row.tune_sec_per_trial,
                row.status,
            )
        return table


def _format_tuning_progress(completed_trials: int, total_trials: int) -> str:
    """Format tuning progress as completed vs total Optuna trials."""
    return f"{completed_trials} / {total_trials}"


def _format_evaluation_progress(completed_rows: int, total_rows: int) -> str:
    """Format row-based evaluation progress with a whole-percent suffix."""
    if total_rows <= 0:
        return "0 / 0 (0%)"
    clipped_rows = min(completed_rows, total_rows)
    percent = int(100 * clipped_rows / total_rows)
    return f"{clipped_rows} / {total_rows} ({percent}%)"


def _progress_table_widths(
    model_names: list[str],
    n_trials: int,
    max_rows: int,
) -> _ProgressTableWidths:
    """Return pinned widths for all rich benchmark progress columns."""
    return _ProgressTableWidths(
        model=max(len("Model"), *(len(model_name) for model_name in model_names)),
        split=max(len("Split"), len("baseline"), len("shifted"), len("evaluating")),
        progress=max(
            len("Progress"),
            len(_format_tuning_progress(n_trials, n_trials)),
            len(_format_evaluation_progress(max_rows, max_rows)),
        ),
        loss=max(len("Last Loss"), len("Best Loss"), BENCHMARK_FLOAT_DISPLAY_WIDTH),
        tune_sec_per_trial=max(len("Tune Sec/Trial"), BENCHMARK_FLOAT_DISPLAY_WIDTH),
        status=max(len("Status"), len("evaluating"), len("evaluated")),
    )


def _format_float_cell(
    value: float | None,
    width: int = BENCHMARK_FLOAT_DISPLAY_WIDTH,
) -> str:
    """Format one float cell with fixed width and no scientific notation."""
    if value is None or not np.isfinite(value):
        return "--".rjust(width)
    absolute_value = abs(value)
    overflow_cell = (("<" if value < 0 else ">") + ("9" * (width - 1))).rjust(width)
    for decimals in range(6, -1, -1):
        formatted = f"{value:.{decimals}f}"
        if len(formatted) <= width:
            return formatted.rjust(width)
    if len(str(int(absolute_value))) >= width:
        return overflow_cell
    return f"{value:.0f}".rjust(width)[:width]


def _format_loss(value: float | None) -> str:
    """Format one optional loss value for the live benchmark table."""
    return _format_float_cell(value)


def _format_tune_sec_per_trial(elapsed_seconds: float, completed_trials: int) -> str:
    """Format the model-local wall-clock tuning time per completed trial."""
    if completed_trials <= 0:
        return "--".rjust(BENCHMARK_FLOAT_DISPLAY_WIDTH)
    return f"{elapsed_seconds / completed_trials:>{BENCHMARK_FLOAT_DISPLAY_WIDTH}.2f}"


def _should_use_rich_progress(stream: object | None = None) -> bool:
    """Return whether the benchmark should render the live rich progress table."""
    output_stream = sys.stdout if stream is None else stream
    isatty = getattr(output_stream, "isatty", None)
    return bool(callable(isatty) and isatty())


def _build_benchmark_progress(
    model_names: list[str],
    n_trials: int,
    max_rows: int,
    enabled: bool | None = None,
) -> _BenchmarkProgress:
    """Build either the live rich controller or a no-op progress controller."""
    progress_enabled = _should_use_rich_progress() if enabled is None else enabled
    if not progress_enabled:
        return _NullBenchmarkProgress()
    try:
        return _RichBenchmarkProgress(
            model_names,
            _progress_table_widths(model_names, n_trials=n_trials, max_rows=max_rows),
        )
    except ImportError:
        return _NullBenchmarkProgress()
