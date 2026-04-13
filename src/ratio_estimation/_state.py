"""Helpers for compact model state snapshots."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np


def serialize_state_value(value: Any) -> Any:
    """Convert common numeric state values into JSON-friendly Python objects."""
    if isinstance(value, dict):
        return {key: serialize_state_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [serialize_state_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    return value


def state_snapshot(**state: Any) -> dict[str, Any]:
    """Build a lightweight serializable state dictionary."""
    return {name: serialize_state_value(value) for name, value in state.items()}
