"""Small Optuna entrypoints for the experiment layer."""

import argparse
from collections.abc import Callable
from typing import cast

import numpy as np
import optuna
import pandas as pd

from ratio_estimation.models import RatioProximalLearner, SoftplusLink

from .data import generate_dataset
from .evaluate import StreamingModel, run_panel


def objective(
    trial: optuna.Trial,
    frame: pd.DataFrame,
    model_builder: Callable[[optuna.Trial], StreamingModel],
    input_column: str = "features",
) -> float:
    """Evaluate one Optuna trial on a panel dataset."""
    mean_loss, stderr = cast(
        tuple[float, float],
        run_panel(
            frame,
            model_factory=lambda: model_builder(trial),
            input_column=input_column,
            return_stderr=True,
        ),
    )
    trial.set_user_attr("stderr", stderr)
    return mean_loss


def tune_model(
    frame: pd.DataFrame,
    model_builder: Callable[[optuna.Trial], StreamingModel],
    n_trials: int = 20,
    seed: int = 0,
    input_column: str = "features",
) -> optuna.Study:
    """Tune one model family on a panel dataset."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, frame, model_builder, input_column=input_column),
        n_trials=n_trials,
    )
    return study


def build_ratio_proximal_model(trial: optuna.Trial) -> RatioProximalLearner:
    """Build the default proximal ratio learner for tuning."""
    step_size = trial.suggest_float("step_size", 1e-8, 100.0, log=True)
    regularization = trial.suggest_float("regularization", 1e-10, 100.0, log=True)
    return RatioProximalLearner(
        link=SoftplusLink(),
        step_size=step_size,
        regularization=regularization,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the tuning entrypoint."""
    parser = argparse.ArgumentParser(
        description="Tune the proximal ratio learner on a synthetic panel."
    )
    parser.add_argument("--groups", type=int, default=20, help="Number of ad groups in the panel.")
    parser.add_argument(
        "--history",
        type=int,
        default=6,
        help="Number of lagged ratio features per example.",
    )
    parser.add_argument("--trials", type=int, default=5, help="Number of Optuna trials to run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dataset generation.")
    return parser.parse_args()


def main() -> None:
    """Run a configurable tuning example from the command line."""
    args = parse_args()
    dataset = generate_dataset(
        n_groups=args.groups,
        history_length=args.history,
        rng=np.random.default_rng(args.seed),
    )
    study = tune_model(
        dataset,
        build_ratio_proximal_model,
        n_trials=args.trials,
        seed=args.seed,
    )
    best_stderr = study.best_trial.user_attrs.get("stderr", float("nan"))
    print(f"groups={args.groups} history={args.history} trials={args.trials} seed={args.seed}")
    print("best value:", study.best_value)
    print("best stderr:", best_stderr)
    print("best params:", study.best_params)


if __name__ == "__main__":
    main()
