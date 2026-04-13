"""Shared JSON and artifact helpers for experiment workflows."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def make_json_safe(value: Any) -> Any:
    """Convert arrays and NumPy scalars into JSON-friendly Python values."""
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def timestamped_output_dir(root: str | Path, prefix: str) -> Path:
    """Return a timestamped artifact directory rooted under one path."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(root) / f"{prefix}-{timestamp}"


def write_json_artifact(path: Path, value: Any) -> None:
    """Write one JSON artifact with the repo-standard formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_json_safe(value), indent=2))


def write_dataframe_artifacts(
    output_dir: Path,
    stem: str,
    frame: pd.DataFrame,
    array_columns: tuple[str, ...] = (),
) -> None:
    """Write one DataFrame as paired CSV and JSON artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{stem}.csv", index=False)
    json_frame = frame if not array_columns else frame.copy()
    for column in array_columns:
        json_frame[column] = json_frame[column].map(list)
    json_frame.to_json(output_dir / f"{stem}.json", orient="records", indent=2)
