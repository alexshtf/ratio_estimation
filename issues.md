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

## 2. High: the experiment generator does not match its implied target semantics

- Files:
  - `experiments/data.py:10-32`
  - `experiments/data.py:35-85`
- Problem:
  - `bounded_periodic_series(...)` is described as bounded between `lower` and `upper`, but the added Gaussian noise can push samples outside that range.
  - `sample_ad_group(...)` generates `true_ratio` first, then sets `count = int(spend / true_ratio)`.
  - Because `count` is floored to an integer, the realized observed ratio `spend / count` is not equal to `true_ratio`.
  - When `count` floors to zero, finite `true_ratio` values create effectively infinite observed ratios.
- Impact:
  - The experiment data does not cleanly represent a stochastic process with a well-defined realized conditional ratio around `true_ratio`.
  - Zero-count rows are introduced by rounding, not only by intended sampling noise.

## 3. Medium-high: decay baselines under-decay across sparse gaps and large jumps

- File:
  - `experiments/baselines.py:363-405`
- Problem:
  - Intended semantics: decay should happen once per elapsed interval, not merely once per arriving event.
  - `DecayRatioBaseline.update(...)` applies at most one decay per observation.
  - In COST and COUNT modes, once the accumulator crosses the threshold, the code decays once and resets the tracked interval to zero, even if several threshold buckets were crossed inside that one update.
  - In TIME mode, if a large time gap crosses several intervals with no events in between, the next event still triggers only one decay.
- Impact:
  - The implemented dynamics are weaker than the intended "once per interval" decay process.
  - Sparse events retain stale history too strongly because missed hourly or interval decays are never applied.
  - Large COST or COUNT jumps also retain too much history because only one decay is applied no matter how many buckets were crossed.

## 4. Medium: public `LinearInverseRatioLearner` can emit infinite or invalid predictions

- File:
  - `src/ratio_estimation/models.py:160-190`
- Problem:
  - `predict(...)` returns `1 / dot(weights, x)`.
  - With zero-initialized weights, the first prediction is `inf` for any nonzero feature vector.
  - Negative or zero scores lead to undefined or sign-invalid ratio predictions.
- Impact:
  - The stable public library exposes a learner that can violate the positive-ratio contract immediately.
  - There is no test coverage for this class.

## 5. Medium-low: `inverse_softplus_normalizer` is a documentation and naming mismatch, not necessarily a math bug

- Files:
  - `src/ratio_estimation/features.py:54-58`
  - `src/ratio_estimation/features.py:94-120`
  - `docs/algorithms.md:81-87`
- Problem:
  - The implementation uses additive smoothing:
    - `ratio = (1 + numerator) / (1 + denominator)`
    - `value = log(expm1(ratio))`
  - That smoothing is intentional and defensible because the raw ratio can be singular or non-finite when `denominator = 0`, `numerator = 0`, or both.
  - The issue is that the function name and docs make it sound like the exact inverse-softplus transform of the raw ratio, which it is not.
- Impact:
  - Readers can easily infer the wrong mathematical meaning from the current API name and documentation.
  - The maintained default feature pipeline is better described as a smoothed inverse-softplus-style normalizer rather than the exact inverse-softplus of `numerator / denominator`.

## 6. Medium: zero/zero observations are scored with an arbitrary large penalty

- Files:
  - `experiments/evaluate.py:79-84`
  - `experiments/evaluate.py:106`
  - `src/ratio_estimation/simulation.py:113-119`
- Problem:
  - `log_ratio_error(...)` uses `nextafter` on both numerator and denominator.
  - For `(numerator, denominator) = (0, 0)`, the observed ratio is effectively treated as `1`.
  - Unless the prediction is also exactly `1`, the loss is very large.
- Impact:
  - Undefined-ratio rows are not handled explicitly.
  - The benchmark metric can be influenced by these arbitrary conventions.

## 7. Medium-low: weighted standard error becomes `nan` for a single retained sample

- File:
  - `experiments/evaluate.py:162-179`
- Problem:
  - `weighted_mean_and_stderr(...)` divides by `n_samples - 1`.
  - `summarize_panel_losses(...)` guards the empty case, but not the single-sample case.
- Impact:
  - Tiny panels or large warmup settings can produce `(mean, nan)` plus a runtime warning.
  - This can propagate into trial metadata and benchmark summaries.

## 8. Low: the progress-table float formatter is not truly fixed-width

- File:
  - `experiments/benchmark.py:262-292`
- Problem:
  - `f"{value:>{width}.6f}"` uses `width` as a minimum width, not a hard cap.
  - Values with four or more integer digits exceed the intended 10-character width.
- Impact:
  - The live benchmark table can still resize despite the fixed-width intent.

## Coverage Gaps

- Missing direct tests for:
  - `LinearInverseRatioLearner`
  - `DecayRatioBaseline`
  - `bounded_periodic_series(...)` bounds semantics
  - `sample_ad_group(...)` zero-count and realized-ratio behavior
  - single-sample behavior in `weighted_mean_and_stderr(...)`

## Check Status At Audit Time

- `uv run ruff check .`: passed
- `uv run pyright`: passed
- `uv run pytest -q`: passed
