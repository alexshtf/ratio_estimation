import json
from pathlib import Path

import numpy as np

from experiments.benchmark import build_benchmark_specs, run_benchmark
from experiments.compare import compare_models
from experiments.data import generate_dataset
from experiments.evaluate import diagnose_stream, rollout_stream, run_panel, score_stream_tail
from experiments.single_stream import (
    build_single_stream_specs,
    generate_single_stream,
    run_single_stream_experiment,
)
from experiments.tune import (
    build_ratio_proximal_model,
    build_ratio_proximal_spec,
    tune_model,
    tune_spec,
)
from ratio_estimation.models import RatioProximalLearner, SoftplusLink


def test_generate_dataset_has_expected_columns() -> None:
    dataset = generate_dataset(n_groups=4, history_length=3, rng=np.random.default_rng(0))
    assert {"id", "offset", "spend", "count", "true_ratio", "features"} <= set(dataset.columns)


def test_generate_single_stream_has_expected_columns() -> None:
    stream = generate_single_stream(history_length=5, rng=np.random.default_rng(0))
    assert {"id", "offset", "spend", "count", "true_ratio", "features"} <= set(stream.columns)
    assert set(stream["id"]) == {0}


def test_shared_registry_preserves_benchmark_and_single_stream_membership() -> None:
    benchmark_names = [spec.name for spec in build_benchmark_specs(history_length=4)]
    single_stream_names = list(build_single_stream_specs(history_length=4))
    assert benchmark_names == single_stream_names[:-1]
    assert single_stream_names[-1] == "exponential_quadratic"


def test_rollout_stream_returns_prediction_frame() -> None:
    dataset = generate_dataset(n_groups=1, history_length=3, rng=np.random.default_rng(0))
    learner = RatioProximalLearner(link=SoftplusLink(), step_size=0.1, regularization=1.0)
    rollout = rollout_stream(dataset, learner)
    assert {"prediction", "actual_ratio", "true_ratio", "log_error"} <= set(rollout.columns)


def test_generate_dataset_matches_benchmark_rolling_feature_semantics() -> None:
    dataset = generate_dataset(n_groups=1, history_length=3, rng=np.random.default_rng(0))
    first_group = dataset[dataset["id"] == 0].reset_index(drop=True)
    spend = np.asarray(first_group["spend"], dtype=float)
    count = np.asarray(first_group["count"], dtype=float)
    share = spend / (spend + count)
    expected_features = np.stack(
        [
            np.pad(
                share[max(0, index - 2) : index + 1],
                (max(0, 2 - index), 0),
            )
            for index in range(len(first_group))
        ]
    )
    for index, expected_feature in enumerate(expected_features):
        np.testing.assert_allclose(first_group.loc[index, "features"], expected_feature)


def test_run_panel_smoke() -> None:
    dataset = generate_dataset(n_groups=6, history_length=4, rng=np.random.default_rng(0))
    mean_loss = run_panel(
        dataset,
        model_factory=lambda: RatioProximalLearner(
            link=SoftplusLink(),
            step_size=0.1,
            regularization=1.0,
        ),
    )
    assert np.isfinite(mean_loss)


def test_compare_models_smoke() -> None:
    scores = compare_models(
        groups=6,
        history_length=4,
        seed=0,
        step_size=0.1,
        regularization=1.0,
    )
    assert set(scores.index) == {
        "proximal_softplus",
        "linear_ratio",
        "linear_regression",
        "ratio_of_regressors",
    }
    assert np.isfinite(scores.to_numpy()).all()


def test_tune_model_smoke() -> None:
    dataset = generate_dataset(n_groups=4, history_length=3, rng=np.random.default_rng(0))
    study = tune_model(dataset, build_ratio_proximal_model, n_trials=2, seed=0)
    assert len(study.trials) == 2


def test_tune_spec_smoke() -> None:
    dataset = generate_dataset(n_groups=4, history_length=3, rng=np.random.default_rng(0))
    study = tune_spec(dataset, build_ratio_proximal_spec(), n_trials=2, seed=0)
    assert len(study.trials) == 2


def test_diagnose_stream_returns_trace_and_state() -> None:
    dataset = generate_dataset(n_groups=1, history_length=3, rng=np.random.default_rng(0))
    learner = RatioProximalLearner(link=SoftplusLink(), step_size=0.1, regularization=1.0)
    diagnostics = diagnose_stream(dataset, learner)
    assert {
        "prediction",
        "actual_ratio",
        "true_ratio",
        "log_error",
    } <= set(diagnostics.trace.columns)
    assert "weights" in diagnostics.final_state
    assert np.isfinite(diagnostics.tail_mean_log_error())


def test_score_stream_tail_matches_trace_tail_loss() -> None:
    dataset = generate_dataset(n_groups=1, history_length=3, rng=np.random.default_rng(0))
    tail_fraction = 0.6
    diagnostics = diagnose_stream(
        dataset,
        RatioProximalLearner(link=SoftplusLink(), step_size=0.1, regularization=1.0),
    )
    tail_score = score_stream_tail(
        dataset,
        RatioProximalLearner(link=SoftplusLink(), step_size=0.1, regularization=1.0),
        tail_fraction=tail_fraction,
    )
    np.testing.assert_allclose(
        tail_score,
        diagnostics.tail_mean_log_error(tail_fraction=tail_fraction),
    )


def test_run_benchmark_writes_expected_artifacts(tmp_path: Path) -> None:
    result = run_benchmark(
        n_trials=1,
        history_length=4,
        tune_groups=8,
        test_groups=12,
        seed=0,
        output_dir=tmp_path,
    )

    assert {
        "model",
        "tune_loss",
        "tune_stderr",
        "same_loss",
        "same_stderr",
        "shifted_loss",
        "shifted_stderr",
    } <= set(result.summary.columns)
    assert len(result.summary) == 8
    assert (result.output_dir / "summary.csv").exists()
    assert (result.output_dir / "summary.json").exists()
    assert (result.output_dir / "best_params.json").exists()
    assert (result.output_dir / "metadata.json").exists()

    metadata = json.loads((result.output_dir / "metadata.json").read_text())
    assert metadata["seed"] == 0


def test_run_single_stream_experiment_writes_expected_artifacts(tmp_path: Path) -> None:
    result = run_single_stream_experiment(
        model_names=["quadratic"],
        history_length=5,
        n_trials=1,
        seed=0,
        output_dir=tmp_path,
        mean_samples=16,
        max_time_offset=20,
    )

    assert {"model", "tail_loss", "full_mean_loss", "steps"} <= set(result.summary.columns)
    assert len(result.summary) == 1
    assert (result.output_dir / "stream.csv").exists()
    assert (result.output_dir / "stream.json").exists()
    assert (result.output_dir / "summary.csv").exists()
    assert (result.output_dir / "best_params.json").exists()
    assert (result.output_dir / "metadata.json").exists()
    assert (result.output_dir / "traces" / "quadratic.csv").exists()
    assert (result.output_dir / "states" / "quadratic.json").exists()
