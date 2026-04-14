import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

import experiments.benchmark as benchmark_module
import experiments.data as data_module
from experiments.baselines import CampaignRunningRatioBaseline
from experiments.benchmark import (
    BASELINE_MODEL_NAME,
    BENCHMARK_FLOAT_DISPLAY_WIDTH,
    _format_evaluation_progress,
    _format_tune_sec_per_trial,
    _format_tuning_progress,
    _progress_table_widths,
    _ProgressRowState,
    _should_use_rich_progress,
    _weighted_rec_curve,
    build_benchmark_specs,
    run_benchmark,
)
from experiments.compare import compare_models
from experiments.data import (
    _sample_ad_group_latent_paths,
    add_autoregressive_features,
    bounded_periodic_series,
    generate_dataset,
    sample_ad_group,
)
from experiments.evaluate import (
    PanelLossSamples,
    diagnose_stream,
    log_ratio_error,
    panel_loss_samples,
    rollout_stream,
    run_panel,
    score_stream_tail,
    summarize_panel_losses,
)
from experiments.plot_groups import _observed_ratio_series, run_plot_groups
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


class _FakeStream:
    def __init__(self, is_tty: bool) -> None:
        self.is_tty = is_tty

    def isatty(self) -> bool:
        return self.is_tty


class _RecordingProgress:
    def __init__(self, model_names: list[str], enabled: bool = True) -> None:
        self.enabled = enabled
        self.rows = {
            model_name: _ProgressRowState(model=model_name) for model_name in model_names
        }
        self.history: list[tuple[str, str, str, str]] = []

    def __enter__(self) -> "_RecordingProgress":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        _ = exc_type, exc, traceback

    def update_row(self, model_name: str, **changes: str) -> None:
        row = self.rows[model_name]
        for field_name, value in changes.items():
            setattr(row, field_name, value)
        self.history.append((model_name, row.split, row.progress, row.status))


def test_generate_dataset_has_expected_columns() -> None:
    dataset = generate_dataset(n_groups=4, history_length=3, rng=np.random.default_rng(0))
    assert {"id", "offset", "spend", "count", "true_ratio", "features"} <= set(dataset.columns)


def test_generate_single_stream_has_expected_columns() -> None:
    stream = generate_single_stream(history_length=5, rng=np.random.default_rng(0))
    assert {"id", "offset", "spend", "count", "true_ratio", "features"} <= set(stream.columns)
    assert set(stream["id"]) == {0}


def test_bounded_periodic_series_stays_within_declared_bounds() -> None:
    series = bounded_periodic_series(
        2.0,
        5.0,
        n_samples=256,
        frequencies=np.array([1, 3]),
        phases=np.array([0.2, 1.4]),
        noise_scale=0.4,
        rng=np.random.default_rng(0),
    )
    assert np.all(series >= 2.0)
    assert np.all(series <= 5.0)


def test_sample_ad_group_latent_paths_are_reproducible_and_ratio_consistent() -> None:
    first = _sample_ad_group_latent_paths(rng=np.random.default_rng(0))
    second = _sample_ad_group_latent_paths(rng=np.random.default_rng(0))

    np.testing.assert_array_equal(first.offset_series, second.offset_series)
    np.testing.assert_allclose(first.spend_mean, second.spend_mean)
    np.testing.assert_allclose(first.count_mean, second.count_mean)
    np.testing.assert_allclose(first.true_ratio, second.true_ratio)
    np.testing.assert_allclose(first.true_ratio, first.spend_mean / first.count_mean)


def test_sample_ad_group_uses_observation_samplers_instead_of_flooring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latent_paths = data_module._LatentAdGroupPaths(
        offset_series=np.array([5, 6], dtype=int),
        spend_mean=np.array([10.0, 20.0]),
        count_mean=np.array([2.0, 4.0]),
        true_ratio=np.array([5.0, 5.0]),
    )

    monkeypatch.setattr(
        data_module,
        "_sample_ad_group_latent_paths",
        lambda *args, **kwargs: latent_paths,
    )
    monkeypatch.setattr(
        data_module,
        "sample_negative_binomial",
        lambda mean, dispersion=0.75, rng=None: np.array([250, 0], dtype=int),
    )
    monkeypatch.setattr(
        data_module,
        "sample_poisson",
        lambda mean, rng=None: np.array([7, 8], dtype=int),
    )

    frame = sample_ad_group(group_id=3, spend_resolution=25, rng=np.random.default_rng(0))

    np.testing.assert_array_equal(frame["offset"].to_numpy(), np.array([5, 6]))
    np.testing.assert_allclose(frame["spend"].to_numpy(dtype=float), np.array([10.0, 0.0]))
    np.testing.assert_array_equal(frame["count"].to_numpy(dtype=int), np.array([7, 8]))
    np.testing.assert_allclose(frame["true_ratio"].to_numpy(dtype=float), np.array([5.0, 5.0]))
    assert not np.array_equal(
        frame["count"].to_numpy(dtype=int),
        np.asarray(latent_paths.spend_mean / latent_paths.true_ratio, dtype=int),
    )


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


def test_log_ratio_error_clips_nonpositive_predictions() -> None:
    assert np.isfinite(log_ratio_error(prediction=-1.0, numerator=2.0, denominator=1.0))


def test_generate_dataset_matches_benchmark_rolling_feature_semantics() -> None:
    dataset = generate_dataset(n_groups=1, history_length=3, rng=np.random.default_rng(0))
    first_group = dataset[dataset["id"] == 0].reset_index(drop=True)
    spend = np.asarray(first_group["spend"], dtype=float)
    count = np.asarray(first_group["count"], dtype=float)
    share = spend / (spend + count)
    expected_features = np.stack(
        [
            np.pad(
                share[max(0, index - 3) : index],
                (max(0, 3 - index), 0),
            )
            for index in range(len(first_group))
        ]
    )
    for index, expected_feature in enumerate(expected_features):
        np.testing.assert_allclose(first_group.loc[index, "features"], expected_feature)


def test_add_autoregressive_features_does_not_leak_the_current_observation() -> None:
    frame = (
        generate_dataset(n_groups=1, history_length=2, rng=np.random.default_rng(0))
        .head(3)
        .copy()
    )
    leaked_variant = frame.copy()
    leaked_variant.loc[1, ["spend", "count"]] = [1000.0, 1.0]

    original = add_autoregressive_features(frame, history_length=2)
    updated = add_autoregressive_features(leaked_variant, history_length=2)

    np.testing.assert_allclose(original.loc[1, "features"], updated.loc[1, "features"])
    assert not np.allclose(original.loc[2, "features"], updated.loc[2, "features"])


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


def test_panel_loss_samples_match_run_panel_summary() -> None:
    dataset = generate_dataset(n_groups=6, history_length=4, rng=np.random.default_rng(0))
    mean_loss, stderr = cast(
        tuple[float, float],
        run_panel(
            dataset,
            model_factory=lambda: RatioProximalLearner(
                link=SoftplusLink(),
                step_size=0.1,
                regularization=1.0,
            ),
            return_stderr=True,
        ),
    )
    samples = panel_loss_samples(
        dataset,
        model_factory=lambda: RatioProximalLearner(
            link=SoftplusLink(),
            step_size=0.1,
            regularization=1.0,
        ),
    )
    sample_mean_loss, sample_stderr = summarize_panel_losses(samples)
    np.testing.assert_allclose(sample_mean_loss, mean_loss)
    np.testing.assert_allclose(sample_stderr, stderr)


def test_campaign_running_ratio_baseline_uses_cumulative_history() -> None:
    baseline = CampaignRunningRatioBaseline()
    assert baseline.predict() == 1.0
    baseline.update(x=0.0, numerator=10.0, denominator=2.0)
    assert baseline.predict() == 5.0
    baseline.update(x=0.0, numerator=6.0, denominator=3.0)
    np.testing.assert_allclose(baseline.predict(), 16.0 / 5.0)


def test_weighted_rec_curve_is_monotone_and_ends_at_one() -> None:
    curve = _weighted_rec_curve(
        PanelLossSamples(
            weights=np.array([1.0, 2.0, 1.0]),
            losses=np.array([0.3, 0.1, 0.3]),
        )
    )
    np.testing.assert_allclose(curve.error_thresholds, np.array([0.0, 0.1, 0.3]))
    np.testing.assert_allclose(curve.cdf, np.array([0.0, 0.5, 1.0]))
    assert np.all(np.diff(curve.error_thresholds) >= 0.0)
    assert np.all(np.diff(curve.cdf) >= 0.0)


def test_benchmark_progress_row_defaults() -> None:
    row = _ProgressRowState(model="quadratic")
    assert row.split == "pending"
    assert row.progress == "--"
    assert row.last_loss == "--"
    assert row.best_loss == "--"
    assert row.tune_sec_per_trial == "--"
    assert row.status == "waiting"


def test_benchmark_progress_helpers_format_expected_strings() -> None:
    assert _format_tuning_progress(3, 10) == "3 / 10"
    assert _format_evaluation_progress(42, 100) == "42 / 100 (42%)"
    assert _format_tune_sec_per_trial(0.57, 3) == f"{0.19:>{BENCHMARK_FLOAT_DISPLAY_WIDTH}.2f}"
    assert len(_format_tune_sec_per_trial(0.57, 3)) == BENCHMARK_FLOAT_DISPLAY_WIDTH


def test_benchmark_progress_widths_cover_max_rendered_content() -> None:
    widths = _progress_table_widths(
        model_names=["quadratic", BASELINE_MODEL_NAME],
        n_trials=100,
        max_rows=20_000,
    )
    assert widths.model >= len(BASELINE_MODEL_NAME)
    assert widths.progress >= len("20000 / 20000 (100%)")
    assert widths.loss == BENCHMARK_FLOAT_DISPLAY_WIDTH
    assert widths.tune_sec_per_trial >= len("Tune Sec/Trial")


def test_benchmark_progress_auto_enable_uses_tty() -> None:
    assert _should_use_rich_progress(_FakeStream(is_tty=True))
    assert not _should_use_rich_progress(_FakeStream(is_tty=False))


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
    assert BASELINE_MODEL_NAME in set(result.summary["model"])
    assert len(result.summary) == 9
    assert (result.output_dir / "summary.csv").exists()
    assert (result.output_dir / "summary.json").exists()
    assert (result.output_dir / "best_params.json").exists()
    assert (result.output_dir / "metadata.json").exists()
    assert result.report_path == result.output_dir / "report.html"
    assert result.report_path.exists()

    metadata = json.loads((result.output_dir / "metadata.json").read_text())
    assert metadata["seed"] == 0
    assert metadata["artifacts"]["report_html"] == str(result.report_path)

    report_html = result.report_path.read_text()
    assert "<svg" in report_html
    assert "campaign_running_ratio" in report_html


def test_run_benchmark_reports_progress_transitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_model_name = build_benchmark_specs(history_length=4)[0].name
    recorder: _RecordingProgress | None = None

    def build_progress(
        model_names: list[str],
        n_trials: int,
        max_rows: int,
        enabled: bool | None = None,
    ) -> _RecordingProgress:
        del n_trials, max_rows, enabled
        nonlocal recorder
        recorder = _RecordingProgress(model_names)
        return recorder

    monkeypatch.setattr(benchmark_module, "_build_benchmark_progress", build_progress)
    run_benchmark(
        n_trials=1,
        history_length=4,
        tune_groups=8,
        test_groups=12,
        seed=0,
        output_dir=tmp_path,
    )

    assert recorder is not None
    assert any(
        model_name == first_model_name
        and split == "tuning"
        and progress == "1 / 1"
        and status == "running"
        for model_name, split, progress, status in recorder.history
    )
    assert any(
        model_name == first_model_name
        and split == "same"
        and progress.endswith("(100%)")
        and status == "evaluating"
        for model_name, split, progress, status in recorder.history
    )
    baseline_row = recorder.rows[BASELINE_MODEL_NAME]
    assert baseline_row.split == "baseline"
    assert baseline_row.progress == "done"
    assert baseline_row.status == "evaluated"
    assert baseline_row.tune_sec_per_trial == "--"
    assert baseline_row.last_loss != "--"
    assert baseline_row.best_loss != "--"


def test_observed_ratio_series_masks_zero_count_rows() -> None:
    frame = pd.DataFrame(
        {
            "id": [0, 0, 0],
            "offset": [0, 1, 2],
            "spend": [6.0, 4.0, 9.0],
            "count": [3, 0, 6],
            "true_ratio": [2.0, 4.0, 1.5],
        }
    )
    observed_ratio = _observed_ratio_series(frame)
    np.testing.assert_allclose(observed_ratio[[0, 2]], np.array([2.0, 1.5]))
    assert np.isnan(observed_ratio[1])


def test_run_plot_groups_writes_html_report(tmp_path: Path) -> None:
    result = run_plot_groups(n_groups=2, seed=0, output_dir=tmp_path)

    assert len(result.groups) == 2
    assert result.report_path == result.output_dir / "report.html"
    assert result.report_path.exists()
    assert (result.output_dir / "metadata.json").exists()

    metadata = json.loads((result.output_dir / "metadata.json").read_text())
    assert metadata["seed"] == 0
    assert metadata["groups"] == 2
    assert metadata["artifacts"]["report_html"] == str(result.report_path)
    assert "zero_spend_rows" in metadata["campaign_summaries"][0]

    report_html = result.report_path.read_text()
    assert "<svg" in report_html
    assert "Campaign 0" in report_html
    assert "zero_spend_rows=" in report_html
    assert "Observed Ratio is shown only where count &gt; 0." in report_html


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
