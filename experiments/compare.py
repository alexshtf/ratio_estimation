"""Quick side-by-side comparisons for the main experiment baselines."""

import argparse
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd

from .data import generate_dataset
from .evaluate import StreamingModel, run_panel
from .registry import comparison_model_factories


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the comparison runner."""
    parser = argparse.ArgumentParser(
        description="Compare a few ratio estimators on one synthetic panel."
    )
    parser.add_argument("--groups", type=int, default=25, help="Number of ad groups in the panel.")
    parser.add_argument(
        "--history",
        type=int,
        default=6,
        help="Number of lagged ratio features per example.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dataset generation.")
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.1,
        help="Default step size for the online models.",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0,
        help="Default regularization strength for the online models.",
    )
    return parser.parse_args()


def build_model_factories(
    history_length: int,
    step_size: float,
    regularization: float,
) -> dict[str, Callable[[], StreamingModel]]:
    """Build the default model comparison registry."""
    return cast(
        dict[str, Callable[[], StreamingModel]],
        comparison_model_factories(history_length, step_size, regularization),
    )


def compare_models(
    groups: int,
    history_length: int,
    seed: int,
    step_size: float,
    regularization: float,
) -> pd.Series:
    """Evaluate the default model registry on one synthetic panel."""
    dataset = generate_dataset(
        n_groups=groups,
        history_length=history_length,
        rng=np.random.default_rng(seed),
    )
    factories = build_model_factories(history_length, step_size, regularization)
    scores = {
        model_name: run_panel(dataset, model_factory=model_factory)
        for model_name, model_factory in factories.items()
    }
    return pd.Series(scores, name="weighted_mean_log_error").sort_values()


def main() -> None:
    """Run the comparison entrypoint from the command line."""
    args = parse_args()
    scores = compare_models(
        groups=args.groups,
        history_length=args.history,
        seed=args.seed,
        step_size=args.step_size,
        regularization=args.regularization,
    )
    print(f"groups={args.groups} history={args.history} seed={args.seed}")
    print(scores.to_string())


if __name__ == "__main__":
    main()
