# Audit Issues

This file records issues found during a code audit on 2026-04-14 and their current status.
Items marked `Resolved` have been fixed on the current branch. Unmarked items remain open.

## 1. Resolved: experiment features no longer leak the current target into the predictor

- Files:
  - `experiments/data.py:88-103`
  - `experiments/benchmark.py:642-665`
  - `experiments/single_stream.py:41-49`
  - `tests/test_experiments_smoke.py:102-118`
- Problem:
  - `add_autoregressive_features(...)` builds each row's feature window from `share[: index + 1]`, so the feature vector includes the current observation.
  - In the experiment layer, `share = spend / (spend + count)` for the current row is available to the model before prediction.
  - For rows with `count > 0`, the observed ratio is a deterministic function of that leaked feature:
    - `ratio = spend / count = share / (1 - share)`
- Impact:
  - Before the fix, the canonical benchmark and single-stream workflows were not causal.
  - Older benchmark artifacts generated before the fix are not directly comparable to new causal runs.
- Notes:
  - The leakage-preserving smoke test was replaced with a causal regression test.

## 2. Resolved: the experiment generator now uses bounded latent means with stochastic observations

- Files:
  - `experiments/data.py:10-32`
  - `experiments/data.py:35-85`
- Problem:
  - The original helper claimed bounded periodic series, but Gaussian perturbations could escape the declared interval.
  - The original generator also set `count = int(spend / true_ratio)`, so `true_ratio` was not the ratio of latent means and zero-count rows came from flooring artifacts.
- Impact:
  - The experiment generator now builds bounded latent `spend_mean` and `true_ratio` paths first, defines `count_mean = spend_mean / true_ratio`, and samples observed count and spend from Poisson and negative-binomial observation models.
  - Experiment `true_ratio` values now have a precise meaning as latent ratios of means, and zero-count rows arise from the observation model rather than deterministic flooring.

## 3. Resolved: decay baselines now apply one decay per elapsed interval

- File:
  - `experiments/baselines.py:363-405`
- Problem:
  - The original baseline applied at most one decay per observation, even when several cost/count buckets or time intervals had elapsed.
- Impact:
  - `DecayRatioBaseline` now applies one multiplicative decay per fully elapsed interval, keeps the leftover remainder for COST and COUNT modes, and applies elapsed TIME decays before incorporating the new observation.
  - Direct regression tests now cover both large COST jumps and sparse TIME gaps.

## 4. Resolved: `LinearInverseRatioLearner` now keeps predictions finite and positive

- File:
  - `src/ratio_estimation/models.py:160-190`
- Problem:
  - The original `predict(...)` returned `1 / dot(weights, x)` directly.
  - With zero-initialized weights, the first prediction was `inf` for any nonzero feature vector.
  - Negative or zero inverse scores also produced invalid ratio predictions.
- Impact:
  - `predict(...)` now clips the inverse score to a tiny positive floor before taking the reciprocal, so the learner satisfies the positive finite ratio contract from cold start onward.
  - Direct regression tests were added for cold-start, negative-score, and stream rollout behavior.

## 5. Resolved: the smoothed inverse-softplus normalizer is now named and documented honestly

- Files:
  - `src/ratio_estimation/features.py:54-58`
  - `src/ratio_estimation/features.py:94-120`
  - `docs/algorithms.md:81-87`
- Problem:
  - The maintained implementation uses additive smoothing:
    - `ratio = (1 + numerator) / (1 + denominator)`
    - `value = log(expm1(ratio))`
  - That smoothing is intentional and necessary because the raw ratio can be singular or non-finite.
  - The old name and docs made it sound like the exact inverse-softplus transform of `numerator / denominator`.
- Impact:
  - The public API now exposes `smoothed_inverse_softplus_normalizer` as the primary name and uses it as the default in `AutoregressiveRatioFeatures`.
  - The legacy `inverse_softplus_normalizer` name remains as a backward-compatible alias, and the docs now describe the smoothing explicitly.

## 6. Resolved: zero/zero observations are treated as undefined and excluded from metrics

- Files:
  - `experiments/evaluate.py:79-84`
  - `experiments/evaluate.py:106`
  - `src/ratio_estimation/simulation.py:113-119`
- Problem:
  - The original metric treated `(numerator, denominator) = (0, 0)` as a finite ratio by nudging both sides with `nextafter`.
- Impact:
  - `log_ratio_error(...)` now returns `NaN` for zero/zero rows, rollout traces mark their `actual_ratio` as missing, and tail/panel summaries skip those undefined rows instead of assigning an arbitrary penalty.

## 7. Resolved: single-sample weighted stderr now returns zero

- File:
  - `experiments/evaluate.py:162-179`
- Problem:
  - The original `weighted_mean_and_stderr(...)` divided by `n_samples - 1` even when only one retained sample remained.
- Impact:
  - One retained sample now returns `(mean, 0.0)`, avoiding runtime warnings and keeping tiny benchmark summaries well-defined.

## 8. Resolved: the progress-table float formatter now stays fixed-width

- File:
  - `experiments/benchmark.py:262-292`
- Problem:
  - The original benchmark formatter used minimum-width float formatting, so large values could still widen the table.
- Impact:
  - The formatter now reduces decimal precision as needed and falls back to a fixed-width overflow sentinel for very large magnitudes, so the live table no longer resizes on numeric growth.

## Coverage Gaps

- Missing direct tests for:
  - none identified in the previously open audited areas

## Check Status At Audit Time

- `uv run ruff check .`: passed
- `uv run pyright`: passed
- `uv run pytest -q`: passed
