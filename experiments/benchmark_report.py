"""REC-curve and HTML-report helpers for benchmark runs."""

from __future__ import annotations

import html
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import PanelLossSamples
from .io import (
    strip_svg_preamble,
    write_dataframe_artifacts,
    write_json_artifact,
)

REPORT_DATASETS = ("tune", "same", "shifted")
REC_LINEAR_WIDTH = 0.01
REC_ZOOM_LINEAR_WIDTH = 1.0
REC_ZOOM_XMAX = 10.0


@dataclass(slots=True)
class _RecCurve:
    """One weighted empirical REC curve."""

    error_thresholds: np.ndarray
    cdf: np.ndarray


def _weighted_rec_curve(samples: PanelLossSamples) -> _RecCurve:
    """Return the weighted empirical REC curve over positive-weight samples."""
    positive_weight_mask = (
        np.isfinite(samples.losses)
        & np.isfinite(samples.weights)
        & (samples.weights > 0.0)
    )
    if not np.any(positive_weight_mask):
        empty = np.empty(0, dtype=float)
        return _RecCurve(error_thresholds=empty, cdf=empty)

    retained_losses = samples.losses[positive_weight_mask]
    retained_weights = samples.weights[positive_weight_mask]
    order = np.argsort(retained_losses, kind="stable")
    sorted_losses = retained_losses[order]
    sorted_weights = retained_weights[order]
    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = float(cumulative_weights[-1])

    is_last_occurrence = np.ones(len(sorted_losses), dtype=bool)
    is_last_occurrence[:-1] = sorted_losses[:-1] != sorted_losses[1:]
    error_thresholds = sorted_losses[is_last_occurrence]
    cdf = cumulative_weights[is_last_occurrence] / total_weight
    start_threshold = 0.0 if error_thresholds[0] > 0.0 else float(error_thresholds[0])
    return _RecCurve(
        error_thresholds=np.concatenate(([start_threshold], error_thresholds)),
        cdf=np.concatenate(([0.0], cdf)),
    )


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
        zoom_axis.set_xlabel(f"Absolute log-ratio error (zoomed to [0, {REC_ZOOM_XMAX:.0f}])")
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
