"""Reusable research helpers for ratio-estimation experiments."""

from .data import add_autoregressive_features, generate_dataset, sample_ad_group
from .evaluate import (
    PanelLossSamples,
    StreamDiagnostics,
    diagnose_stream,
    log_ratio_error,
    panel_loss_samples,
    rollout_stream,
    run_panel,
    summarize_panel_losses,
    tail_mean_log_error,
    weighted_mean_and_stderr,
)

__all__ = [
    "PanelLossSamples",
    "StreamDiagnostics",
    "add_autoregressive_features",
    "diagnose_stream",
    "generate_dataset",
    "log_ratio_error",
    "panel_loss_samples",
    "rollout_stream",
    "run_panel",
    "sample_ad_group",
    "summarize_panel_losses",
    "tail_mean_log_error",
    "weighted_mean_and_stderr",
]
