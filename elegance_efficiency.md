# Elegance And Efficiency Audit

## Overall Assessment

The core library in `src/ratio_estimation` is already in good shape for a research repo:

- `src/ratio_estimation/proximal.py` is compact, mathematically direct, and mostly free of clutter.
- `src/ratio_estimation/models.py` is easy to follow and keeps the online learners small.
- `src/ratio_estimation/simulation.py` is vectorized where it should be.

Most of the remaining elegance and efficiency debt is in the `experiments` layer, especially where sequential workflows are expressed through Pandas-heavy orchestration.

There are no critical problems. The main opportunities are:

1. remove avoidable Python-object materialization in streaming loops
2. vectorize rolling feature construction
3. avoid building full diagnostic DataFrames inside Optuna objectives
4. reduce duplicated experiment orchestration code

## Findings

### 1. Medium: streaming evaluation materializes whole frames as Python dicts

Files:

- `experiments/evaluate.py:79`
- `experiments/evaluate.py:132`

Both `diagnose_stream()` and `run_panel()` iterate through `frame.to_dict(orient="records")`.

That is elegant enough for tiny data, but it is the wrong shape for an inherently sequential numeric loop:

- it allocates a full list of Python dicts up front
- every row lookup becomes repeated string-key dictionary access
- it throws away the main benefit of having numeric columns already stored in arrays

For this codebase, the best refinement would be one of:

- iterate lazily with `frame.itertuples(index=False)`
- or, better, extract the few required columns once as arrays and loop over zipped NumPy/object arrays

This would keep the code just as readable while cutting a clear source of overhead in the hot path.

### 2. Medium: autoregressive feature generation is more Pythonic than necessary

File:

- `experiments/data.py:81`

`add_autoregressive_features()` builds each feature row with a Python list comprehension, repeated slicing, and repeated `np.pad(...)`.

That is readable, but it is one of the least vectorized pieces in the maintained codebase. For a benchmark dataset with many groups, this becomes avoidable overhead.

The natural cleanup would be:

- pre-pad the 1D ratio-share array once
- build the lag matrix in one shot with `np.lib.stride_tricks.sliding_window_view(...)`
- convert rows back to arrays only if the downstream interface truly needs per-row arrays

This would improve both elegance and speed.

### 3. Medium: single-stream tuning builds full diagnostic traces for every trial

Files:

- `experiments/single_stream.py:85`
- `experiments/single_stream.py:112`

The Optuna objective in `tune_single_stream()` calls `evaluate_single_stream()`, which calls `diagnose_stream()`, which builds a full trace `DataFrame` for every trial.

That is substantially more work than the objective needs. The objective only needs one scalar:

- tail mean log error

It does not need:

- the full prediction trace
- the actual-ratio column
- the true-ratio column
- the final state snapshot
- a Pandas frame allocation

This is the single clearest efficiency opportunity in the repo today.

Recommended refinement:

- add a lightweight scalar scorer like `score_stream_tail(...)`
- use it inside the Optuna objective
- call `diagnose_stream()` only once for the best parameters after tuning

That would preserve the current API while making the single-stream sanity workflow much cheaper.

### 4. Low: benchmark and single-stream model registries duplicate orchestration logic

Files:

- `experiments/benchmark.py:54`
- `experiments/single_stream.py:50`

The maintained benchmark suite and the single-stream suite both carry registry-like logic:

- model names
- input-column routing
- parameter search spaces
- model builders

The duplication is still manageable, but it is starting to become structural noise.

This is mostly an elegance issue:

- the benchmark path and single-stream path should feel like two evaluation modes over one shared model registry
- right now they feel like sibling scripts with partially copied configuration logic

Recommended direction:

- extract one shared typed registry module for model specs
- let benchmark and single-stream runners differ only in scoring/output logic

### 5. Low: repeated `state_dict()` boilerplate makes baseline classes noisier than needed

Files:

- `experiments/baselines.py:63`
- `experiments/baselines.py:122`
- `experiments/baselines.py:178`
- `experiments/baselines.py:251`
- `experiments/baselines.py:317`
- `experiments/baselines.py:381`
- `src/ratio_estimation/models.py:113`
- `src/ratio_estimation/models.py:155`
- `src/ratio_estimation/models.py:196`

The state snapshot support is useful, but the repeated dictionary boilerplate adds some visual weight.

This is not a correctness problem, and the current code is still readable. But if the experiment layer grows further, a tiny helper or mixin for common fields would keep the learner/baseline classes more focused on the math.

### 6. Low: rolling means are recomputed from full window histories on every update

File:

- `src/ratio_estimation/features.py:64`

`RollingMeanWindow` stores the full window and calls `np.mean(...)` over the whole buffer each time.

That is acceptable for small windows, and it keeps the code simple. But it is `O(window_size)` per update instead of `O(1)`.

If window sizes stay small, I would leave it alone.
If larger windows become common in experiments, maintaining rolling sums would be the cleaner high-performance version.

### 7. Low: experiment result writing is clear but not especially DRY

Files:

- `experiments/benchmark.py:232`
- `experiments/single_stream.py:126`

The report/artifact writing code is straightforward, but both modules have their own near-parallel file-writing flows.

This is fine for now, but if more experiment entrypoints are added, a shared result-writer utility would reduce noise and make output conventions easier to keep aligned.

## Strong Parts Worth Preserving

- `src/ratio_estimation/proximal.py:13` through `src/ratio_estimation/proximal.py:76`
  - very compact and mathematically legible
- `src/ratio_estimation/models.py:74` through `src/ratio_estimation/models.py:202`
  - small learners, clean separation of link and update logic
- `src/ratio_estimation/simulation.py:27` through `src/ratio_estimation/simulation.py:119`
  - good use of NumPy vectorization in the synthetic generator
- `experiments/evaluate.py:54` through `experiments/evaluate.py:151`
  - the scoring logic is easy to understand, even if the row iteration can be tightened

## Recommended Priority Order

1. Replace `to_dict(orient="records")` loops with tuple or array-based iteration in `experiments/evaluate.py`.
2. Add a scalar-only single-stream tuning objective so Optuna does not build full traces every trial.
3. Vectorize `add_autoregressive_features()` in `experiments/data.py`.
4. Factor the shared model registry between `benchmark.py` and `single_stream.py`.
5. Only then consider smaller cleanliness refactors like shared `state_dict()` helpers.

## Bottom Line

The repo is already fairly clean where it matters most: the math core is concise and readable.

The remaining opportunities are mostly about making the experiment layer less allocation-heavy and less duplicated, without pushing the repo toward production-style abstraction.
