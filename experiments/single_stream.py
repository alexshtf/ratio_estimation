"""Single-stream tuning and diagnostics for quick research sanity checks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from .data import add_autoregressive_features, sample_ad_group
from .evaluate import StreamDiagnostics, diagnose_stream, score_stream_tail
from .io import (
    make_json_safe,
    timestamped_output_dir,
    write_dataframe_artifacts,
    write_json_artifact,
)
from .registry import ExperimentModelSpec, single_stream_model_specs


@dataclass(slots=True)
class SingleStreamResult:
    """Structured output from one single-stream tuning run."""

    stream: pd.DataFrame
    summary: pd.DataFrame
    best_params: dict[str, dict[str, Any]]
    metadata: dict[str, Any]
    output_dir: Path


def default_output_dir() -> Path:
    """Return a timestamped default artifact directory for single-stream runs."""
    return timestamped_output_dir("artifacts/single_stream", "single-stream")


def generate_single_stream(
    history_length: int = 5,
    rng: np.random.Generator | None = None,
    **stream_kwargs: Any,
) -> pd.DataFrame:
    """Generate one benchmark-style stream with rolling ratio-share features."""
    generator = np.random.default_rng() if rng is None else rng
    frame = sample_ad_group(group_id=0, rng=generator, **stream_kwargs)
    return add_autoregressive_features(frame, history_length=history_length)


def build_single_stream_specs(history_length: int) -> dict[str, ExperimentModelSpec]:
    """Build the maintained single-stream model registry."""
    return single_stream_model_specs(history_length)


def resolve_model_names(
    specs: dict[str, ExperimentModelSpec],
    model_names: list[str] | tuple[str, ...],
) -> list[str]:
    """Resolve user-supplied model names, expanding `all` when requested."""
    if model_names == ["all"] or model_names == ("all",):
        return list(specs)

    unknown = sorted(set(model_names) - set(specs))
    if unknown:
        available = ", ".join(specs)
        unknown_str = ", ".join(unknown)
        raise ValueError(f"Unknown model(s): {unknown_str}. Available models: {available}.")
    return list(model_names)


def evaluate_single_stream(
    frame: pd.DataFrame,
    spec: ExperimentModelSpec,
    params: dict[str, Any],
    tail_fraction: float = 0.9,
) -> tuple[float, StreamDiagnostics]:
    """Evaluate one tuned model on one stream with the maintained tail-loss objective."""
    diagnostics = diagnose_stream(
        frame,
        spec.build_model(params),
        input_column=spec.input_column,
    )
    tail_loss = diagnostics.tail_mean_log_error(tail_fraction=tail_fraction)
    return tail_loss, diagnostics


def tune_single_stream(
    frame: pd.DataFrame,
    spec: ExperimentModelSpec,
    n_trials: int = 50,
    seed: int = 0,
    tail_fraction: float = 0.9,
) -> optuna.Study:
    """Tune one model family on one stream using the maintained tail-loss objective."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = spec.suggest_params(trial)
        return score_stream_tail(
            frame,
            spec.build_model(params),
            input_column=spec.input_column,
            tail_fraction=tail_fraction,
        )

    study.optimize(objective, n_trials=n_trials)
    return study


def write_artifacts(
    output_dir: Path,
    stream: pd.DataFrame,
    summary: pd.DataFrame,
    best_params: dict[str, dict[str, Any]],
    diagnostics_by_model: dict[str, StreamDiagnostics],
    metadata: dict[str, Any],
) -> None:
    """Write the stream, summaries, and per-model diagnostics to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    state_dir = output_dir / "states"
    trace_dir.mkdir(exist_ok=True)
    state_dir.mkdir(exist_ok=True)

    write_dataframe_artifacts(output_dir, "stream", stream, array_columns=("features",))
    write_dataframe_artifacts(output_dir, "summary", summary)
    write_json_artifact(output_dir / "best_params.json", best_params)
    write_json_artifact(output_dir / "metadata.json", metadata)

    for model_name, diagnostics in diagnostics_by_model.items():
        write_dataframe_artifacts(trace_dir, model_name, diagnostics.trace)
        write_json_artifact(state_dir / f"{model_name}.json", diagnostics.final_state)


def run_single_stream_experiment(
    model_names: list[str] | tuple[str, ...] = ("quadratic",),
    history_length: int = 5,
    n_trials: int = 50,
    seed: int = 0,
    tail_fraction: float = 0.9,
    output_dir: str | Path | None = None,
    **stream_kwargs: Any,
) -> SingleStreamResult:
    """Tune and evaluate one or more models on a single benchmark-style stream."""
    output_path = Path(output_dir) if output_dir is not None else default_output_dir()
    master_rng = np.random.default_rng(seed)
    stream_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
    stream = generate_single_stream(
        history_length=history_length,
        rng=np.random.default_rng(stream_seed),
        **stream_kwargs,
    )

    specs = build_single_stream_specs(history_length)
    selected_models = resolve_model_names(specs, model_names)

    rows: list[dict[str, Any]] = []
    best_params: dict[str, dict[str, Any]] = {}
    diagnostics_by_model: dict[str, StreamDiagnostics] = {}
    study_seeds: dict[str, int] = {}

    for model_name in selected_models:
        spec = specs[model_name]
        study_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
        study_seeds[model_name] = study_seed
        study = tune_single_stream(
            stream,
            spec,
            n_trials=n_trials,
            seed=study_seed,
            tail_fraction=tail_fraction,
        )
        params = make_json_safe(study.best_params)
        tail_loss, diagnostics = evaluate_single_stream(
            stream,
            spec,
            params,
            tail_fraction=tail_fraction,
        )
        rows.append(
            {
                "model": model_name,
                "tail_loss": tail_loss,
                "full_mean_loss": float(diagnostics.trace["log_error"].mean()),
                "steps": len(diagnostics.trace),
            }
        )
        best_params[model_name] = {
            "params": params,
            "tail_fraction": tail_fraction,
            "input_column": spec.input_column,
        }
        diagnostics_by_model[model_name] = diagnostics

    summary = pd.DataFrame(rows).sort_values("tail_loss").reset_index(drop=True)
    metadata = {
        "seed": seed,
        "stream_seed": stream_seed,
        "history_length": history_length,
        "n_trials": n_trials,
        "tail_fraction": tail_fraction,
        "models": selected_models,
        "stream_kwargs": stream_kwargs,
        "study_seeds": study_seeds,
        "artifacts": {
            "stream_csv": str(output_path / "stream.csv"),
            "stream_json": str(output_path / "stream.json"),
            "summary_csv": str(output_path / "summary.csv"),
            "summary_json": str(output_path / "summary.json"),
            "best_params_json": str(output_path / "best_params.json"),
            "metadata_json": str(output_path / "metadata.json"),
            "trace_dir": str(output_path / "traces"),
            "state_dir": str(output_path / "states"),
        },
    }
    write_artifacts(
        output_path,
        stream,
        summary,
        best_params,
        diagnostics_by_model,
        metadata,
    )
    return SingleStreamResult(
        stream=stream,
        summary=summary,
        best_params=best_params,
        metadata=make_json_safe(metadata),
        output_dir=output_path,
    )


def format_summary_table(summary: pd.DataFrame) -> str:
    """Format a single-stream summary for terminal display."""
    return summary.to_string(index=False, float_format=lambda value: f"{value:.6f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the single-stream runner."""
    parser = argparse.ArgumentParser(
        description="Run the maintained single-stream sanity-check workflow."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["quadratic"],
        help="Model names to tune, or `all` to run the full single-stream suite.",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=5,
        help="Rolling feature window size for the single stream.",
    )
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per model.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.9,
        help="Fraction of the stream used for the single-stream tail-loss objective.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Artifact directory.")
    parser.add_argument("--mean-spend", type=float, default=100.0, help="Mean spend scale.")
    parser.add_argument("--mean-ratio", type=float, default=5.0, help="Mean ratio scale.")
    parser.add_argument(
        "--mean-samples",
        type=float,
        default=24 * 7,
        help="Mean number of observations in the stream.",
    )
    parser.add_argument(
        "--mean-frequency",
        type=float,
        default=1.5,
        help="Mean latent periodic frequency count.",
    )
    parser.add_argument(
        "--num-frequencies",
        type=int,
        default=2,
        help="Number of latent periodic components.",
    )
    parser.add_argument(
        "--max-time-offset",
        type=int,
        default=24 * 9,
        help="Maximum stream start offset.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the single-stream CLI."""
    args = parse_args()
    result = run_single_stream_experiment(
        model_names=args.models,
        history_length=args.history,
        n_trials=args.trials,
        seed=args.seed,
        tail_fraction=args.tail_fraction,
        output_dir=args.output_dir,
        mean_spend=args.mean_spend,
        mean_ratio=args.mean_ratio,
        mean_samples=args.mean_samples,
        mean_frequency=args.mean_frequency,
        num_frequencies=args.num_frequencies,
        max_time_offset=args.max_time_offset,
    )
    print(f"output_dir={result.output_dir}")
    print(format_summary_table(result.summary))


if __name__ == "__main__":
    main()
