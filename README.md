# ratio_estimation

Research code for online ratio estimation on synthetic campaign time series.

## Layout

- `src/ratio_estimation`: small stable library for simulation, features, proximal math, and ratio learners
- `experiments`: reusable research helpers, baselines, evaluation loops, and Optuna tuning
- `notebooks`: thin demos that import the library and experiment helpers
- `docs`: short human-readable notes, including the algorithm summary
- `artifacts`: generated outputs and legacy study logs

## Setup

```bash
uv sync --all-groups
```

Run the checks with:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

## Library Example

```python
import numpy as np

from ratio_estimation.features import AutoregressiveRatioFeatures
from ratio_estimation.models import RatioProximalLearner, SoftplusLink
from ratio_estimation.simulation import sample_campaign

campaign = sample_campaign(rng=np.random.default_rng(0))
feature_builder = AutoregressiveRatioFeatures(history_length=12)
model = RatioProximalLearner(link=SoftplusLink(), step_size=0.1, regularization=1.0)

predictions = []
for spend, count in zip(campaign.spend, campaign.count, strict=True):
    x = feature_builder.features()
    predictions.append(model.predict(x))
    model.update(x, numerator=spend, denominator=count)
    feature_builder.update(spend, count)
```

## Experiment Examples

```bash
uv run python -m experiments.tune
```

The canonical benchmark workflow is:

```bash
uv run python -m experiments.benchmark
```

The cheap single-stream sanity-check workflow is:

```bash
uv run python -m experiments.single_stream --models quadratic
```

Or use the `Makefile` wrappers:

```bash
make compare
make tune TRIALS=50 GROUPS=40 HISTORY=8 SEED=1
make benchmark TRIALS=100 HISTORY=4 SEED=0 TUNE_GROUPS=1000 TEST_GROUPS=20000
make stream MODELS=quadratic TRIALS=50 HISTORY=5 SEED=0 TAIL_FRACTION=0.9
```

`make benchmark` prints the same-distribution vs shifted-distribution table and writes
`summary.csv`, `summary.json`, `best_params.json`, and `metadata.json` under
`artifacts/benchmarks/...`.

`make stream` tunes one or more models on a single synthetic stream, writes the stream,
per-model traces, final states, and summary files under `artifacts/single_stream/...`, and
uses the maintained tail-loss sanity objective.

The maintained notebooks are demos. Reusable models, evaluation logic, and tuning code belong in
Python modules, not in notebook cells.
