"""Visualize synthetic experiment groups for generator inspection."""

from __future__ import annotations

import argparse
import html
import inspect
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import sample_ad_group
from .io import make_json_safe, strip_svg_preamble, timestamped_output_dir, write_json_artifact


@dataclass(slots=True)
class PlotGroupsResult:
    """Structured output from one synthetic-group plotting run."""

    groups: list[pd.DataFrame]
    metadata: dict[str, Any]
    output_dir: Path
    report_path: Path


def default_output_dir() -> Path:
    """Return the default artifact directory for synthetic-group plots."""
    return timestamped_output_dir("artifacts/plot_groups", "plot-groups")


def _group_parameter_defaults() -> dict[str, Any]:
    """Return the maintained default generator parameters for one synthetic group."""
    signature = inspect.signature(sample_ad_group)
    excluded = {"group_id", "rng"}
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if name not in excluded and parameter.default is not inspect._empty
    }


def _observed_ratio_series(frame: pd.DataFrame) -> np.ndarray:
    """Return the realized ratio, masking zero-count rows as missing values."""
    spend = frame["spend"].to_numpy(dtype=float, copy=False)
    count = frame["count"].to_numpy(dtype=float, copy=False)
    observed_ratio = np.full(len(frame), np.nan, dtype=float)
    nonzero_mask = count > 0.0
    observed_ratio[nonzero_mask] = spend[nonzero_mask] / count[nonzero_mask]
    return observed_ratio


def _campaign_summary_lines(frame: pd.DataFrame) -> list[str]:
    """Summarize one campaign for the HTML report."""
    offsets = frame["offset"].to_numpy(dtype=int, copy=False)
    zero_spend_rows = int((frame["spend"].to_numpy(dtype=float, copy=False) == 0.0).sum())
    zero_count_rows = int((frame["count"].to_numpy(dtype=int, copy=False) == 0).sum())
    return [
        f"campaign_id={int(frame['id'].iloc[0])}",
        f"rows={len(frame)}",
        f"offset_range=[{int(offsets.min())}, {int(offsets.max())}]",
        f"zero_spend_rows={zero_spend_rows}",
        f"zero_count_rows={zero_count_rows}",
    ]


def _render_campaign_figure_svg(frame: pd.DataFrame) -> str:
    """Render one campaign as a 1x3 SVG figure."""
    import matplotlib.pyplot as plt

    offsets = frame["offset"].to_numpy(dtype=float, copy=False)
    spend = frame["spend"].to_numpy(dtype=float, copy=False)
    count = frame["count"].to_numpy(dtype=float, copy=False)
    true_ratio = frame["true_ratio"].to_numpy(dtype=float, copy=False)
    observed_ratio = _observed_ratio_series(frame)
    zero_count_mask = np.isnan(observed_ratio)

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 3.8), constrained_layout=True)
    campaign_id = int(frame["id"].iloc[0])
    figure.suptitle(f"Campaign {campaign_id}")

    axes[0].plot(offsets, spend, color="tab:blue")
    axes[0].set_title("Spend")
    axes[0].set_xlabel("Offset")
    axes[0].set_ylabel("Numerator")

    axes[1].plot(offsets, count, color="tab:orange")
    axes[1].set_title("Count")
    axes[1].set_xlabel("Offset")
    axes[1].set_ylabel("Denominator")

    axes[2].plot(offsets, true_ratio, color="tab:green", label="True Ratio")
    axes[2].plot(offsets, observed_ratio, color="tab:purple", label="Observed Ratio")
    if np.any(zero_count_mask):
        axes[2].scatter(
            offsets[zero_count_mask],
            true_ratio[zero_count_mask],
            color="tab:red",
            marker="x",
            label="count == 0",
        )
    axes[2].set_title("Ratio")
    axes[2].set_xlabel("Offset")
    axes[2].set_ylabel("Ratio")
    axes[2].legend(loc="best")

    buffer = StringIO()
    figure.savefig(buffer, format="svg")
    plt.close(figure)
    return strip_svg_preamble(buffer.getvalue())


def _report_run_lines(metadata: dict[str, Any]) -> list[str]:
    """Build the plain-text run metadata block for the HTML report."""
    generator_parameters = metadata["generator_parameters"]
    lines = [
        f"seed={metadata['seed']}",
        f"groups={metadata['groups']}",
    ]
    lines.extend(f"{key}={value}" for key, value in generator_parameters.items())
    return lines


def _render_report_html(groups: list[pd.DataFrame], metadata: dict[str, Any]) -> str:
    """Render the plain HTML plot-groups report."""
    run_lines = html.escape("\n".join(_report_run_lines(metadata)))
    sections: list[str] = []

    for frame in groups:
        campaign_id = int(frame["id"].iloc[0])
        campaign_summary = html.escape("\n".join(_campaign_summary_lines(frame)))
        sections.extend(
            [
                f"<h2>Campaign {campaign_id}</h2>",
                f"<pre>{campaign_summary}</pre>",
                (
                    "<p>True Ratio is the latent ratio of means for the campaign. "
                    "Observed Ratio is shown only where count &gt; 0. "
                    "Rows with count == 0 are marked on the True Ratio curve.</p>"
                ),
                _render_campaign_figure_svg(frame),
            ]
        )

    return "\n".join(
        [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            "<title>Generated Campaign Plots</title>",
            "</head>",
            "<body>",
            "<h1>Generated Campaign Plots</h1>",
            "<h2>Run</h2>",
            f"<pre>{run_lines}</pre>",
            *sections,
            "</body>",
            "</html>",
        ]
    )


def _write_artifacts(output_dir: Path, metadata: dict[str, Any], report_html: str) -> None:
    """Write the plot-groups HTML report and metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_artifact(output_dir / "metadata.json", metadata)
    (output_dir / "report.html").write_text(report_html, encoding="utf-8")


def run_plot_groups(
    n_groups: int = 6,
    seed: int = 0,
    output_dir: str | Path | None = None,
    **group_kwargs: Any,
) -> PlotGroupsResult:
    """Generate and plot several synthetic campaigns under the current experiment generator."""
    output_path = Path(output_dir) if output_dir is not None else default_output_dir()
    generator = np.random.default_rng(seed)
    generator_parameters = _group_parameter_defaults() | group_kwargs
    groups = [
        sample_ad_group(group_id=group_id, rng=generator, **group_kwargs)
        for group_id in range(n_groups)
    ]
    report_path = output_path / "report.html"
    metadata = {
        "seed": seed,
        "groups": n_groups,
        "generator_parameters": generator_parameters,
        "campaign_summaries": [
            {
                "campaign_id": int(frame["id"].iloc[0]),
                "rows": len(frame),
                "offset_start": int(frame["offset"].min()),
                "offset_end": int(frame["offset"].max()),
                "zero_spend_rows": int((frame["spend"] == 0.0).sum()),
                "zero_count_rows": int((frame["count"] == 0).sum()),
            }
            for frame in groups
        ],
        "artifacts": {
            "report_html": str(report_path),
            "metadata_json": str(output_path / "metadata.json"),
        },
    }
    report_html = _render_report_html(groups, metadata)
    _write_artifacts(output_path, metadata, report_html)
    return PlotGroupsResult(
        groups=groups,
        metadata=make_json_safe(metadata),
        output_dir=output_path,
        report_path=report_path,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for synthetic-group plotting."""
    parser = argparse.ArgumentParser(
        description="Plot several generated campaigns from the current experiment generator."
    )
    parser.add_argument("--groups", type=int, default=6, help="Number of campaigns to plot.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Artifact directory.")
    return parser.parse_args()


def main() -> None:
    """Run the plot-groups CLI."""
    args = parse_args()
    result = run_plot_groups(
        n_groups=args.groups,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"output_dir={result.output_dir}")
    print(f"report_html={result.report_path}")


if __name__ == "__main__":
    main()
