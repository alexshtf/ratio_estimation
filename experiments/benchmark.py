"""Benchmark workflow for same-distribution and shifted-distribution tables."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import optuna
import pandas as pd

from .data import generate_dataset
from .evaluate import StreamingModel, run_panel
from .io import (
    make_json_safe,
    timestamped_output_dir,
    write_dataframe_artifacts,
    write_json_artifact,
)
from .registry import ExperimentModelSpec, benchmark_model_specs


@dataclass(slots=True)
class BenchmarkResult:
    """Structured output from one full benchmark run."""

    summary: pd.DataFrame
    best_params: dict[str, dict[str, Any]]
    metadata: dict[str, Any]
    output_dir: Path


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
            return_stderr=True,
        ),
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


def write_artifacts(
    output_dir: Path,
    summary: pd.DataFrame,
    best_params: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    """Write benchmark artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe_artifacts(output_dir, "summary", summary)
    write_json_artifact(output_dir / "best_params.json", best_params)
    write_json_artifact(output_dir / "metadata.json", metadata)


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
    """Tune the benchmark suite and evaluate same and shifted tables."""
    output_path = Path(output_dir) if output_dir is not None else default_output_dir()
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

    for spec in specs:
        study_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
        study_seeds[spec.name] = study_seed
        params, tune_loss, tune_stderr = tune_spec(
            tune_frame,
            spec,
            n_trials=n_trials,
            seed=study_seed,
        )
        same_loss, same_stderr = evaluate_spec(same_frame, spec, params)
        shifted_loss, shifted_stderr = evaluate_spec(shifted_frame, spec, params)
        rows.append(
            {
                "model": spec.name,
                "tune_loss": tune_loss,
                "tune_stderr": tune_stderr,
                "same_loss": same_loss,
                "same_stderr": same_stderr,
                "shifted_loss": shifted_loss,
                "shifted_stderr": shifted_stderr,
            }
        )
        best_params[spec.name] = {
            "params": params,
            "tune_loss": tune_loss,
            "tune_stderr": tune_stderr,
            "input_column": spec.input_column,
        }

    summary = pd.DataFrame(rows).sort_values("same_loss").reset_index(drop=True)
    metadata = {
        "seed": seed,
        "n_trials": n_trials,
        "history_length": history_length,
        "tune_groups": tune_groups,
        "test_groups": test_groups,
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
        },
    }
    write_artifacts(output_path, summary, best_params, metadata)
    return BenchmarkResult(
        summary=summary,
        best_params=best_params,
        metadata=make_json_safe(metadata),
        output_dir=output_path,
    )


def format_summary_table(summary: pd.DataFrame) -> str:
    """Format a benchmark summary for terminal display."""
    return summary.to_string(index=False, float_format=lambda value: f"{value:.6f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Tune and evaluate the maintained same-vs-shifted benchmark suite."
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
    print(format_summary_table(result.summary))


if __name__ == "__main__":
    main()
