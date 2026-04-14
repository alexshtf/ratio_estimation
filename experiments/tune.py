"""Small Optuna entrypoints for the experiment layer."""

import argparse
from collections.abc import Callable
from typing import cast

import numpy as np
import optuna
import pandas as pd

from ratio_estimation.models import RatioProximalLearner

from .data import generate_dataset
from .evaluate import StreamingModel, run_panel
from .registry import ExperimentModelSpec, proximal_softplus_spec


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


def tune_spec(
    frame: pd.DataFrame,
    spec: ExperimentModelSpec,
    n_trials: int = 20,
    seed: int = 0,
) -> optuna.Study:
    """Tune one experiment spec on a panel dataset."""
    return tune_model(
        frame,
        model_builder=lambda trial: cast(
            StreamingModel,
            spec.build_model(spec.suggest_params(trial)),
        ),
        n_trials=n_trials,
        seed=seed,
        input_column=spec.input_column,
    )


def build_ratio_proximal_spec() -> ExperimentModelSpec:
    """Return the maintained tuning spec for the softplus proximal learner."""
    return proximal_softplus_spec()


def build_ratio_proximal_model(trial: optuna.Trial) -> RatioProximalLearner:
    """Build the default proximal ratio learner for tuning."""
    spec = build_ratio_proximal_spec()
    return cast(
        RatioProximalLearner,
        spec.build_model(spec.suggest_params(trial)),
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
        help="Number of previous ratio-share observations per example.",
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
    study = tune_spec(
        dataset,
        build_ratio_proximal_spec(),
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
