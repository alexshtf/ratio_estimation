# AGENTS

## Purpose

This repo is a research codebase for online ratio estimation on synthetic campaign time series. The code should stay mathematically clear, compact, and easy to modify.

## Structure

- `src/ratio_estimation`: stable library code
- `experiments`: reusable research flows, baselines, evaluation, tuning
- `notebooks`: thin demos only
- `docs`: short project documentation
- `artifacts`: generated outputs and archived historical logs

## Style

- Add short docstrings to every public function and class.
- Prefer vectorized NumPy and small readable helpers.
- Prefer clear names over short names.
- Do not add defensive boilerplate unless it makes the math easier to follow.
- Keep the public package small and coherent.
- Do not put reusable logic in notebooks.

## Workflow

- Use `uv run` for commands.
- Keep `ruff`, `pyright`, and `pytest` green.
- Update `README.md` and `docs/algorithms.md` when public APIs or algorithm descriptions change.
- Extend `experiments/` instead of growing notebook-local code.
- Treat `experiments.benchmark` and `make benchmark` as the canonical archived research workflow.
