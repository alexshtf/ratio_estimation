"""Reusable research helpers for ratio-estimation experiments."""

from .data import add_autoregressive_features, generate_dataset, sample_ad_group
from .evaluate import (
    StreamDiagnostics,
    diagnose_stream,
    log_ratio_error,
    rollout_stream,
    run_panel,
    tail_mean_log_error,
    weighted_mean_and_stderr,
)

__all__ = [
    "StreamDiagnostics",
    "add_autoregressive_features",
    "diagnose_stream",
    "generate_dataset",
    "log_ratio_error",
    "rollout_stream",
    "run_panel",
    "sample_ad_group",
    "tail_mean_log_error",
    "weighted_mean_and_stderr",
]
