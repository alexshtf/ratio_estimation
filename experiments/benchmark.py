"""Benchmark workflow for same-distribution summaries and an REC HTML report."""

from __future__ import annotations

import argparse
import html
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import optuna
import pandas as pd

from .baselines import CampaignRunningRatioBaseline
from .data import generate_dataset
from .evaluate import (
    PanelLossSamples,
    PanelProgressCallback,
    StreamingModel,
    panel_loss_samples,
    run_panel,
    summarize_panel_losses,
)
from .io import (
    make_json_safe,
    strip_svg_preamble,
    timestamped_output_dir,
    write_dataframe_artifacts,
    write_json_artifact,
)
from .registry import ExperimentModelSpec, benchmark_model_specs

if TYPE_CHECKING:
    from rich.console import RenderableType

BASELINE_MODEL_NAME = "campaign_running_ratio"
BASELINE_INPUT_COLUMN = "offset"
BASELINE_DEFAULT_PREDICTION = 1.0
BENCHMARK_WARMUP_STEPS = 2
BENCHMARK_EVALUATION_PROGRESS_FREQUENCY = 1_000
BENCHMARK_FLOAT_DISPLAY_WIDTH = 10
REPORT_DATASETS = ("tune", "same", "shifted")
REC_LINEAR_WIDTH = 0.01
REC_ZOOM_LINEAR_WIDTH = 1.0
REC_ZOOM_XMAX = 10.0

type BenchmarkModelSpec = ExperimentModelSpec
type TuneProgressCallback = Callable[[int, int, float, float, float], None]


@dataclass(slots=True)
class BenchmarkResult:
    """Structured output from one full benchmark run."""

    summary: pd.DataFrame
    best_params: dict[str, dict[str, Any]]
    metadata: dict[str, Any]
    output_dir: Path
    report_path: Path


@dataclass(slots=True)
class _RecCurve:
    error_thresholds: np.ndarray
    cdf: np.ndarray


@dataclass(slots=True)
class _PanelEvaluation:
    mean_loss: float
    stderr: float
    rec_curve: _RecCurve


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


def default_output_dir() -> Path:
    """Return a timestamped default artifact directory."""
    return timestamped_output_dir("artifacts/benchmarks", "benchmark")


def build_benchmark_specs(history_length: int) -> list[BenchmarkModelSpec]:
    """Build the maintained benchmark model suite."""
    return benchmark_model_specs(history_length)


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


def _format_float_cell(value: float | None, width: int = BENCHMARK_FLOAT_DISPLAY_WIDTH) -> str:
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


def evaluate_spec(
    frame: pd.DataFrame,
    spec: BenchmarkModelSpec,
    params: dict[str, Any],
) -> tuple[float, float]:
    """Evaluate one benchmark model on one panel."""
    return cast(
        tuple[float, float],
        run_panel(
            frame,
            model_factory=lambda: cast(StreamingModel, spec.build_model(params)),
            input_column=spec.input_column,
            warmup_steps=BENCHMARK_WARMUP_STEPS,
            return_stderr=True,
        ),
    )


def _weighted_rec_curve(samples: PanelLossSamples) -> _RecCurve:
    """Return the weighted empirical REC curve for one retained panel sample set."""
    if len(samples.losses) == 0:
        empty = np.empty(0, dtype=float)
        return _RecCurve(error_thresholds=empty, cdf=empty)

    order = np.argsort(samples.losses, kind="stable")
    sorted_losses = samples.losses[order]
    sorted_weights = samples.weights[order]
    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = float(cumulative_weights[-1])
    if total_weight <= 0.0:
        empty = np.empty(0, dtype=float)
        return _RecCurve(error_thresholds=empty, cdf=empty)

    is_last_occurrence = np.ones(len(sorted_losses), dtype=bool)
    is_last_occurrence[:-1] = sorted_losses[:-1] != sorted_losses[1:]
    error_thresholds = sorted_losses[is_last_occurrence]
    cdf = cumulative_weights[is_last_occurrence] / total_weight
    start_threshold = 0.0 if error_thresholds[0] > 0.0 else float(error_thresholds[0])
    return _RecCurve(
        error_thresholds=np.concatenate(([start_threshold], error_thresholds)),
        cdf=np.concatenate(([0.0], cdf)),
    )


def _evaluate_panel(
    frame: pd.DataFrame,
    model_factory: Callable[[], StreamingModel],
    input_column: str,
    progress_callback: PanelProgressCallback | None = None,
) -> _PanelEvaluation:
    """Evaluate one panel and return both summary metrics and the REC curve."""
    samples = panel_loss_samples(
        frame,
        model_factory=model_factory,
        input_column=input_column,
        warmup_steps=BENCHMARK_WARMUP_STEPS,
        progress_callback=progress_callback,
        progress_frequency=BENCHMARK_EVALUATION_PROGRESS_FREQUENCY,
    )
    mean_loss, stderr = summarize_panel_losses(samples)
    return _PanelEvaluation(
        mean_loss=mean_loss,
        stderr=stderr,
        rec_curve=_weighted_rec_curve(samples),
    )


def _evaluate_tuned_spec(
    frame: pd.DataFrame,
    spec: BenchmarkModelSpec,
    params: dict[str, Any],
    progress_callback: PanelProgressCallback | None = None,
) -> _PanelEvaluation:
    """Evaluate one tuned benchmark model and build its REC curve."""
    return _evaluate_panel(
        frame,
        model_factory=lambda: cast(StreamingModel, spec.build_model(params)),
        input_column=spec.input_column,
        progress_callback=progress_callback,
    )


def _evaluate_baseline(
    frame: pd.DataFrame,
    progress_callback: PanelProgressCallback | None = None,
) -> _PanelEvaluation:
    """Evaluate the untuned campaign running-ratio baseline on one panel."""
    return _evaluate_panel(
        frame,
        model_factory=lambda: CampaignRunningRatioBaseline(
            default_prediction=BASELINE_DEFAULT_PREDICTION
        ),
        input_column=BASELINE_INPUT_COLUMN,
        progress_callback=progress_callback,
    )


def tune_spec(
    frame: pd.DataFrame,
    spec: BenchmarkModelSpec,
    n_trials: int,
    seed: int,
    progress_callback: TuneProgressCallback | None = None,
) -> tuple[dict[str, Any], float, float]:
    """Tune one benchmark model family on one panel."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    started_at = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        params = spec.suggest_params(trial)
        mean_loss, stderr = evaluate_spec(frame, spec, params)
        trial.set_user_attr("stderr", stderr)
        return mean_loss

    callbacks: list[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]] = []
    if progress_callback is not None:

        def report_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            last_value = np.nan if trial.value is None else trial.value
            progress_callback(
                trial.number + 1,
                n_trials,
                float(last_value),
                float(study.best_value),
                time.perf_counter() - started_at,
            )

        callbacks.append(report_trial)

    study.optimize(objective, n_trials=n_trials, callbacks=callbacks or None)
    best_trial = study.best_trial
    best_stderr = best_trial.user_attrs.get("stderr", np.nan)
    best_value = np.nan if best_trial.value is None else best_trial.value
    return make_json_safe(best_trial.params), float(best_value), float(best_stderr)


def _report_run_lines(metadata: dict[str, Any]) -> list[str]:
    """Return the plain-text benchmark metadata shown in the HTML report."""
    train_dataset = metadata["train_dataset"]
    same_dataset = metadata["same_dataset"]
    shifted_dataset = metadata["shifted_dataset"]
    return [
        f"seed={metadata['seed']}",
        f"history_length={metadata['history_length']}",
        f"n_trials={metadata['n_trials']}",
        f"tune_groups={metadata['tune_groups']}",
        f"test_groups={metadata['test_groups']}",
        (
            "train_dataset: "
            f"mean_spend={train_dataset['mean_spend']} "
            f"mean_ratio={train_dataset['mean_ratio']} "
            f"max_time_offset={train_dataset['max_time_offset']} "
            f"seed={train_dataset['seed']}"
        ),
        (
            "same_dataset: "
            f"mean_spend={same_dataset['mean_spend']} "
            f"mean_ratio={same_dataset['mean_ratio']} "
            f"max_time_offset={same_dataset['max_time_offset']} "
            f"seed={same_dataset['seed']}"
        ),
        (
            "shifted_dataset: "
            f"mean_spend={shifted_dataset['mean_spend']} "
            f"mean_ratio={shifted_dataset['mean_ratio']} "
            f"max_time_offset={shifted_dataset['max_time_offset']} "
            f"seed={shifted_dataset['seed']}"
        ),
    ]


def _render_rec_figure_svg(
    curves_by_dataset: dict[str, dict[str, _RecCurve]],
    model_order: list[str],
) -> str:
    """Render overview and zoomed REC panels for the three benchmark splits."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(15.0, 8.6), sharey=True)
    figure.suptitle("Benchmark REC Curves")
    panel_titles = {
        "tune": "Tune",
        "same": "Same",
        "shifted": "Shifted",
    }

    for column_index, dataset_name in enumerate(REPORT_DATASETS):
        overview_axis = axes[0, column_index]
        zoom_axis = axes[1, column_index]
        for model_name in model_order:
            curve = curves_by_dataset[dataset_name][model_name]
            if len(curve.error_thresholds) == 0:
                continue
            overview_axis.step(
                curve.error_thresholds,
                curve.cdf,
                where="post",
                label=model_name,
            )
            zoom_axis.step(
                curve.error_thresholds,
                curve.cdf,
                where="post",
                label=model_name,
            )
        overview_axis.set_title(f"{panel_titles[dataset_name]} REC")
        overview_axis.set_xscale("asinh", linear_width=REC_LINEAR_WIDTH)
        overview_axis.set_xlabel("Absolute log-ratio error (asinh scale)")
        overview_axis.set_ylim(0.0, 1.0)

        zoom_axis.set_title(f"{panel_titles[dataset_name]} REC Zoom")
        zoom_axis.set_xscale("asinh", linear_width=REC_ZOOM_LINEAR_WIDTH)
        zoom_axis.set_xlim(0.0, REC_ZOOM_XMAX)
        zoom_axis.set_xlabel("Absolute log-ratio error (zoomed to [0, 10])")
        zoom_axis.set_ylim(0.0, 1.0)

    axes[0, 0].set_ylabel("Spend-weighted CDF")
    axes[1, 0].set_ylabel("Spend-weighted CDF")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.subplots_adjust(bottom=0.18, top=0.90, hspace=0.45, wspace=0.2)

    buffer = StringIO()
    figure.savefig(buffer, format="svg")
    plt.close(figure)
    return strip_svg_preamble(buffer.getvalue())


def _render_report_html(summary: pd.DataFrame, metadata: dict[str, Any], svg_markup: str) -> str:
    """Render the plain HTML benchmark report with one embedded SVG figure."""
    run_lines = html.escape("\n".join(_report_run_lines(metadata)))
    summary_html = summary.to_html(
        index=False,
        border=1,
        float_format=lambda value: f"{value:.6f}",
    )
    baseline_text = html.escape(
        "campaign_running_ratio predicts each campaign's cumulative spend/count ratio "
        "before the current update, with 1.0 as the zero-history fallback."
    )
    return "\n".join(
        [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            "<title>Benchmark Report</title>",
            "</head>",
            "<body>",
            "<h1>Benchmark Report</h1>",
            "<h2>Run</h2>",
            f"<pre>{run_lines}</pre>",
            "<h2>Summary</h2>",
            summary_html,
            "<h2>REC Curves</h2>",
            f"<p>{baseline_text}</p>",
            svg_markup,
            "</body>",
            "</html>",
        ]
    )


def write_artifacts(
    output_dir: Path,
    summary: pd.DataFrame,
    best_params: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    report_html: str,
) -> None:
    """Write benchmark artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe_artifacts(output_dir, "summary", summary)
    write_json_artifact(output_dir / "best_params.json", best_params)
    write_json_artifact(output_dir / "metadata.json", metadata)
    (output_dir / "report.html").write_text(report_html, encoding="utf-8")


def _run_evaluation_with_progress(
    progress: _BenchmarkProgress,
    model_name: str,
    split: str,
    total_rows: int,
    evaluate_panel: Callable[[PanelProgressCallback], _PanelEvaluation],
) -> _PanelEvaluation:
    """Run one evaluation split while streaming row progress into the live table."""
    progress.update_row(
        model_name,
        split=split,
        progress=_format_evaluation_progress(0, total_rows),
        status="evaluating",
    )

    def report_rows(completed_rows: int, total_rows_in_callback: int) -> None:
        progress.update_row(
            model_name,
            progress=_format_evaluation_progress(completed_rows, total_rows_in_callback),
        )

    return evaluate_panel(report_rows)


def run_benchmark(
    n_trials: int = 100,
    history_length: int = 4,
    tune_groups: int = 1000,
    test_groups: int = 20000,
    seed: int = 0,
    output_dir: str | Path | None = None,
    train_mean_spend: float = 100.0,
    train_mean_ratio: float = 5.0,
    shift_mean_spend: float = 90.0,
    shift_mean_ratio: float = 6.0,
    train_max_time_offset: int = 24 * 9,
    test_max_time_offset: int = 24 * 20,
) -> BenchmarkResult:
    """Tune the benchmark suite, keep the summary table, and build the REC report."""
    output_path = Path(output_dir) if output_dir is not None else default_output_dir()
    report_path = output_path / "report.html"
    master_rng = np.random.default_rng(seed)
    tune_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
    same_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
    shifted_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))

    tune_frame = generate_dataset(
        n_groups=tune_groups,
        history_length=history_length,
        mean_spend=train_mean_spend,
        mean_ratio=train_mean_ratio,
        max_time_offset=train_max_time_offset,
        rng=np.random.default_rng(tune_seed),
    )
    same_frame = generate_dataset(
        n_groups=test_groups,
        history_length=history_length,
        mean_spend=train_mean_spend,
        mean_ratio=train_mean_ratio,
        max_time_offset=test_max_time_offset,
        rng=np.random.default_rng(same_seed),
    )
    shifted_frame = generate_dataset(
        n_groups=test_groups,
        history_length=history_length,
        mean_spend=shift_mean_spend,
        mean_ratio=shift_mean_ratio,
        max_time_offset=test_max_time_offset,
        rng=np.random.default_rng(shifted_seed),
    )

    rows: list[dict[str, Any]] = []
    best_params: dict[str, dict[str, Any]] = {}
    specs = build_benchmark_specs(history_length)
    study_seeds: dict[str, int] = {}
    curves_by_dataset = {dataset_name: {} for dataset_name in REPORT_DATASETS}
    progress = _build_benchmark_progress(
        [spec.name for spec in specs] + [BASELINE_MODEL_NAME],
        n_trials=n_trials,
        max_rows=max(len(tune_frame), len(same_frame), len(shifted_frame)),
    )
    previous_optuna_verbosity = optuna.logging.get_verbosity() if progress.enabled else None
    if previous_optuna_verbosity is not None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        with progress:
            for spec in specs:
                progress.update_row(
                    spec.name,
                    split="tuning",
                    progress=_format_tuning_progress(0, n_trials),
                    status="running",
                )

                def report_tuning(
                    completed_trials: int,
                    total_trials: int,
                    last_loss: float,
                    best_loss: float,
                    elapsed_seconds: float,
                    model_name: str = spec.name,
                ) -> None:
                    progress.update_row(
                        model_name,
                        progress=_format_tuning_progress(completed_trials, total_trials),
                        last_loss=_format_loss(last_loss),
                        best_loss=_format_loss(best_loss),
                        tune_sec_per_trial=_format_tune_sec_per_trial(
                            elapsed_seconds,
                            completed_trials,
                        ),
                        status="running",
                    )

                study_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
                study_seeds[spec.name] = study_seed
                params, _, _ = tune_spec(
                    tune_frame,
                    spec,
                    n_trials=n_trials,
                    seed=study_seed,
                    progress_callback=report_tuning,
                )
                tune_evaluation = _run_evaluation_with_progress(
                    progress,
                    spec.name,
                    split="tune",
                    total_rows=len(tune_frame),
                    evaluate_panel=lambda callback, spec=spec, params=params: _evaluate_tuned_spec(
                        tune_frame,
                        spec,
                        params,
                        progress_callback=callback,
                    ),
                )
                same_evaluation = _run_evaluation_with_progress(
                    progress,
                    spec.name,
                    split="same",
                    total_rows=len(same_frame),
                    evaluate_panel=lambda callback, spec=spec, params=params: _evaluate_tuned_spec(
                        same_frame,
                        spec,
                        params,
                        progress_callback=callback,
                    ),
                )
                shifted_evaluation = _run_evaluation_with_progress(
                    progress,
                    spec.name,
                    split="shifted",
                    total_rows=len(shifted_frame),
                    evaluate_panel=lambda callback, spec=spec, params=params: _evaluate_tuned_spec(
                        shifted_frame,
                        spec,
                        params,
                        progress_callback=callback,
                    ),
                )
                progress.update_row(
                    spec.name,
                    progress="done",
                    status="evaluated",
                )
                rows.append(
                    {
                        "model": spec.name,
                        "tune_loss": tune_evaluation.mean_loss,
                        "tune_stderr": tune_evaluation.stderr,
                        "same_loss": same_evaluation.mean_loss,
                        "same_stderr": same_evaluation.stderr,
                        "shifted_loss": shifted_evaluation.mean_loss,
                        "shifted_stderr": shifted_evaluation.stderr,
                    }
                )
                curves_by_dataset["tune"][spec.name] = tune_evaluation.rec_curve
                curves_by_dataset["same"][spec.name] = same_evaluation.rec_curve
                curves_by_dataset["shifted"][spec.name] = shifted_evaluation.rec_curve
                best_params[spec.name] = {
                    "params": params,
                    "tune_loss": tune_evaluation.mean_loss,
                    "tune_stderr": tune_evaluation.stderr,
                    "input_column": spec.input_column,
                }

            baseline_tune = _run_evaluation_with_progress(
                progress,
                BASELINE_MODEL_NAME,
                split="tune",
                total_rows=len(tune_frame),
                evaluate_panel=lambda callback: _evaluate_baseline(
                    tune_frame,
                    progress_callback=callback,
                ),
            )
            baseline_same = _run_evaluation_with_progress(
                progress,
                BASELINE_MODEL_NAME,
                split="same",
                total_rows=len(same_frame),
                evaluate_panel=lambda callback: _evaluate_baseline(
                    same_frame,
                    progress_callback=callback,
                ),
            )
            progress.update_row(
                BASELINE_MODEL_NAME,
                last_loss=_format_loss(baseline_same.mean_loss),
                best_loss=_format_loss(baseline_same.mean_loss),
            )
            baseline_shifted = _run_evaluation_with_progress(
                progress,
                BASELINE_MODEL_NAME,
                split="shifted",
                total_rows=len(shifted_frame),
                evaluate_panel=lambda callback: _evaluate_baseline(
                    shifted_frame,
                    progress_callback=callback,
                ),
            )
            progress.update_row(
                BASELINE_MODEL_NAME,
                split="baseline",
                progress="done",
                status="evaluated",
            )
    finally:
        if previous_optuna_verbosity is not None:
            optuna.logging.set_verbosity(previous_optuna_verbosity)

    rows.append(
        {
            "model": BASELINE_MODEL_NAME,
            "tune_loss": baseline_tune.mean_loss,
            "tune_stderr": baseline_tune.stderr,
            "same_loss": baseline_same.mean_loss,
            "same_stderr": baseline_same.stderr,
            "shifted_loss": baseline_shifted.mean_loss,
            "shifted_stderr": baseline_shifted.stderr,
        }
    )
    curves_by_dataset["tune"][BASELINE_MODEL_NAME] = baseline_tune.rec_curve
    curves_by_dataset["same"][BASELINE_MODEL_NAME] = baseline_same.rec_curve
    curves_by_dataset["shifted"][BASELINE_MODEL_NAME] = baseline_shifted.rec_curve
    best_params[BASELINE_MODEL_NAME] = {
        "params": {},
        "tune_loss": baseline_tune.mean_loss,
        "tune_stderr": baseline_tune.stderr,
        "input_column": BASELINE_INPUT_COLUMN,
    }

    summary = pd.DataFrame(rows).sort_values("same_loss").reset_index(drop=True)
    metadata = {
        "seed": seed,
        "n_trials": n_trials,
        "history_length": history_length,
        "tune_groups": tune_groups,
        "test_groups": test_groups,
        "baseline_model": BASELINE_MODEL_NAME,
        "baseline_default_prediction": BASELINE_DEFAULT_PREDICTION,
        "rec_weighting": "spend",
        "warmup_steps": BENCHMARK_WARMUP_STEPS,
        "train_dataset": {
            "mean_spend": train_mean_spend,
            "mean_ratio": train_mean_ratio,
            "max_time_offset": train_max_time_offset,
            "seed": tune_seed,
        },
        "same_dataset": {
            "mean_spend": train_mean_spend,
            "mean_ratio": train_mean_ratio,
            "max_time_offset": test_max_time_offset,
            "seed": same_seed,
        },
        "shifted_dataset": {
            "mean_spend": shift_mean_spend,
            "mean_ratio": shift_mean_ratio,
            "max_time_offset": test_max_time_offset,
            "seed": shifted_seed,
        },
        "study_seeds": study_seeds,
        "artifacts": {
            "summary_csv": str(output_path / "summary.csv"),
            "summary_json": str(output_path / "summary.json"),
            "best_params_json": str(output_path / "best_params.json"),
            "metadata_json": str(output_path / "metadata.json"),
            "report_html": str(report_path),
        },
    }
    report_html = _render_report_html(
        summary,
        metadata,
        _render_rec_figure_svg(curves_by_dataset, summary["model"].tolist()),
    )
    write_artifacts(output_path, summary, best_params, metadata, report_html)
    return BenchmarkResult(
        summary=summary,
        best_params=best_params,
        metadata=make_json_safe(metadata),
        output_dir=output_path,
        report_path=report_path,
    )


def format_summary_table(summary: pd.DataFrame) -> str:
    """Format a benchmark summary for terminal display."""
    return summary.to_string(index=False, float_format=lambda value: f"{value:.6f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Tune the maintained benchmark suite and write the REC HTML report."
    )
    parser.add_argument("--trials", type=int, default=100, help="Optuna trials per model.")
    parser.add_argument(
        "--history",
        type=int,
        default=4,
        help="Number of previous ratio-share observations per example.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--tune-groups",
        type=int,
        default=1000,
        help="Ad groups in the tuning set.",
    )
    parser.add_argument(
        "--test-groups",
        type=int,
        default=20000,
        help="Ad groups in each test set.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Artifact directory.")
    parser.add_argument(
        "--train-mean-spend",
        type=float,
        default=100.0,
        help="Mean spend for the tuning and same-distribution datasets.",
    )
    parser.add_argument(
        "--train-mean-ratio",
        type=float,
        default=5.0,
        help="Mean ratio for the tuning and same-distribution datasets.",
    )
    parser.add_argument(
        "--shift-mean-spend",
        type=float,
        default=90.0,
        help="Mean spend for the shifted dataset.",
    )
    parser.add_argument(
        "--shift-mean-ratio",
        type=float,
        default=6.0,
        help="Mean ratio for the shifted dataset.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark CLI."""
    args = parse_args()
    result = run_benchmark(
        n_trials=args.trials,
        history_length=args.history,
        tune_groups=args.tune_groups,
        test_groups=args.test_groups,
        seed=args.seed,
        output_dir=args.output_dir,
        train_mean_spend=args.train_mean_spend,
        train_mean_ratio=args.train_mean_ratio,
        shift_mean_spend=args.shift_mean_spend,
        shift_mean_ratio=args.shift_mean_ratio,
    )
    print(f"output_dir={result.output_dir}")
    print(f"report_html={result.report_path}")
    print(format_summary_table(result.summary))


if __name__ == "__main__":
    main()
