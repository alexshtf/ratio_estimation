"""Benchmark workflow for same-distribution summaries and an REC HTML report."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import optuna
import pandas as pd

from .baselines import CampaignRunningRatioBaseline
from .benchmark_progress import (
    _BenchmarkProgress,
    _build_benchmark_progress,
    _format_evaluation_progress,
    _format_loss,
    _format_tune_sec_per_trial,
    _format_tuning_progress,
)
from .benchmark_report import (
    REPORT_DATASETS,
    _RecCurve,
    _render_rec_figure_svg,
    _render_report_html,
    _weighted_rec_curve,
    write_artifacts,
)
from .data import generate_dataset
from .evaluate import (
    PanelProgressCallback,
    StreamingModel,
    panel_loss_samples,
    run_panel,
    summarize_panel_losses,
)
from .io import make_json_safe, timestamped_output_dir
from .registry import ExperimentModelSpec, benchmark_model_specs

BASELINE_MODEL_NAME = "campaign_running_ratio"
BASELINE_INPUT_COLUMN = "offset"
BASELINE_DEFAULT_PREDICTION = 1.0
BENCHMARK_WARMUP_STEPS = 2
BENCHMARK_EVALUATION_PROGRESS_FREQUENCY = 1_000

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
class _PanelEvaluation:
    mean_loss: float
    stderr: float
    rec_curve: _RecCurve


def default_output_dir() -> Path:
    """Return a timestamped default artifact directory."""
    return timestamped_output_dir("artifacts/benchmarks", "benchmark")


def build_benchmark_specs(history_length: int) -> list[BenchmarkModelSpec]:
    """Build the maintained benchmark model suite."""
    return benchmark_model_specs(history_length)


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
