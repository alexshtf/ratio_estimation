"""Benchmark workflow for same-distribution summaries and an REC HTML report."""

from __future__ import annotations

import argparse
import html
from collections.abc import Callable
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import optuna
import pandas as pd

from .baselines import CampaignRunningRatioBaseline
from .data import generate_dataset
from .evaluate import (
    PanelLossSamples,
    StreamingModel,
    panel_loss_samples,
    run_panel,
    summarize_panel_losses,
)
from .io import (
    make_json_safe,
    timestamped_output_dir,
    write_dataframe_artifacts,
    write_json_artifact,
)
from .registry import ExperimentModelSpec, benchmark_model_specs

BASELINE_MODEL_NAME = "campaign_running_ratio"
BASELINE_INPUT_COLUMN = "offset"
BASELINE_DEFAULT_PREDICTION = 1.0
BENCHMARK_WARMUP_STEPS = 2
REPORT_DATASETS = ("tune", "same", "shifted")


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


def default_output_dir() -> Path:
    """Return a timestamped default artifact directory."""
    return timestamped_output_dir("artifacts/benchmarks", "benchmark")


type BenchmarkModelSpec = ExperimentModelSpec


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
) -> _PanelEvaluation:
    """Evaluate one panel and return both summary metrics and the REC curve."""
    samples = panel_loss_samples(
        frame,
        model_factory=model_factory,
        input_column=input_column,
        warmup_steps=BENCHMARK_WARMUP_STEPS,
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
) -> _PanelEvaluation:
    """Evaluate one tuned benchmark model and build its REC curve."""
    return _evaluate_panel(
        frame,
        model_factory=lambda: cast(StreamingModel, spec.build_model(params)),
        input_column=spec.input_column,
    )


def _evaluate_baseline(frame: pd.DataFrame) -> _PanelEvaluation:
    """Evaluate the untuned campaign running-ratio baseline on one panel."""
    return _evaluate_panel(
        frame,
        model_factory=lambda: CampaignRunningRatioBaseline(
            default_prediction=BASELINE_DEFAULT_PREDICTION
        ),
        input_column=BASELINE_INPUT_COLUMN,
    )


def tune_spec(
    frame: pd.DataFrame,
    spec: BenchmarkModelSpec,
    n_trials: int,
    seed: int,
) -> tuple[dict[str, Any], float, float]:
    """Tune one benchmark model family on one panel."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = spec.suggest_params(trial)
        mean_loss, stderr = evaluate_spec(frame, spec, params)
        trial.set_user_attr("stderr", stderr)
        return mean_loss

    study.optimize(objective, n_trials=n_trials)
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


def _strip_svg_preamble(svg_markup: str) -> str:
    """Remove the XML preamble so the SVG can be inlined directly into HTML."""
    lines = [
        line
        for line in svg_markup.splitlines()
        if not line.startswith("<?xml") and not line.startswith("<!DOCTYPE")
    ]
    return "\n".join(lines)


def _render_rec_figure_svg(
    curves_by_dataset: dict[str, dict[str, _RecCurve]],
    model_order: list[str],
) -> str:
    """Render the three split REC panels as one inline SVG figure."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    figure.suptitle("Benchmark REC Curves")
    panel_titles = {
        "tune": "Tune",
        "same": "Same",
        "shifted": "Shifted",
    }

    for axis, dataset_name in zip(axes, REPORT_DATASETS, strict=True):
        for model_name in model_order:
            curve = curves_by_dataset[dataset_name][model_name]
            if len(curve.error_thresholds) == 0:
                continue
            axis.step(
                curve.error_thresholds,
                curve.cdf,
                where="post",
                label=model_name,
            )
        axis.set_title(f"{panel_titles[dataset_name]} REC")
        axis.set_xlabel("Absolute log-ratio error")
        axis.set_ylim(0.0, 1.0)

    axes[0].set_ylabel("Spend-weighted CDF")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.subplots_adjust(bottom=0.34, top=0.82, wspace=0.2)

    buffer = StringIO()
    figure.savefig(buffer, format="svg")
    plt.close(figure)
    return _strip_svg_preamble(buffer.getvalue())


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
    curves_by_dataset = {
        dataset_name: {} for dataset_name in REPORT_DATASETS
    }

    for spec in specs:
        study_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
        study_seeds[spec.name] = study_seed
        params, _, _ = tune_spec(
            tune_frame,
            spec,
            n_trials=n_trials,
            seed=study_seed,
        )
        tune_evaluation = _evaluate_tuned_spec(tune_frame, spec, params)
        same_evaluation = _evaluate_tuned_spec(same_frame, spec, params)
        shifted_evaluation = _evaluate_tuned_spec(shifted_frame, spec, params)
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

    baseline_tune = _evaluate_baseline(tune_frame)
    baseline_same = _evaluate_baseline(same_frame)
    baseline_shifted = _evaluate_baseline(shifted_frame)
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
    parser.add_argument("--history", type=int, default=4, help="Rolling feature window size.")
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
