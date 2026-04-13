UV_CACHE_DIR ?= /tmp/uv-cache
UV_RUN = UV_CACHE_DIR=$(UV_CACHE_DIR) uv run

GROUPS ?= 25
HISTORY ?= 6
TRIALS ?= 20
SEED ?= 0
STEP_SIZE ?= 0.1
REGULARIZATION ?= 1.0
TUNE_GROUPS ?= 1000
TEST_GROUPS ?= 20000
MODELS ?= quadratic
TAIL_FRACTION ?= 0.9

.PHONY: help sync lock lint format format-check typecheck test check compare tune benchmark stream lab

help:
	@printf "Available targets:\n"
	@printf "  sync          Install dependencies with uv\n"
	@printf "  lock          Refresh uv.lock\n"
	@printf "  lint          Run ruff checks\n"
	@printf "  format        Format Python and notebook files with ruff\n"
	@printf "  format-check  Check formatting without rewriting\n"
	@printf "  typecheck     Run pyright\n"
	@printf "  test          Run pytest\n"
	@printf "  check         Run lint, format-check, typecheck, and test\n"
	@printf "  compare       Compare baseline experiment runners\n"
	@printf "  tune          Tune the proximal learner with Optuna\n"
	@printf "  benchmark     Run the maintained same-vs-shifted benchmark tables\n"
	@printf "  stream        Run the maintained single-stream sanity-check workflow\n"
	@printf "  lab           Launch JupyterLab\n"
	@printf "\n"
	@printf "Experiment variables:\n"
	@printf "  GROUPS=%s HISTORY=%s TRIALS=%s SEED=%s STEP_SIZE=%s REGULARIZATION=%s TUNE_GROUPS=%s TEST_GROUPS=%s MODELS=%s TAIL_FRACTION=%s\n" \
		"$(GROUPS)" "$(HISTORY)" "$(TRIALS)" "$(SEED)" "$(STEP_SIZE)" "$(REGULARIZATION)" "$(TUNE_GROUPS)" "$(TEST_GROUPS)" "$(MODELS)" "$(TAIL_FRACTION)"

sync:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv sync --all-groups

lock:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv lock

lint:
	$(UV_RUN) ruff check .

format:
	$(UV_RUN) ruff format .

format-check:
	$(UV_RUN) ruff format --check .

typecheck:
	$(UV_RUN) pyright

test:
	$(UV_RUN) pytest

check: lint format-check typecheck test

compare:
	$(UV_RUN) python -m experiments.compare \
		--groups $(GROUPS) \
		--history $(HISTORY) \
		--seed $(SEED) \
		--step-size $(STEP_SIZE) \
		--regularization $(REGULARIZATION)

tune:
	$(UV_RUN) python -m experiments.tune \
		--groups $(GROUPS) \
		--history $(HISTORY) \
		--trials $(TRIALS) \
		--seed $(SEED)

benchmark:
	$(UV_RUN) python -m experiments.benchmark \
		--trials $(TRIALS) \
		--history $(HISTORY) \
		--seed $(SEED) \
		--tune-groups $(TUNE_GROUPS) \
		--test-groups $(TEST_GROUPS)

stream:
	$(UV_RUN) python -m experiments.single_stream \
		--models $(MODELS) \
		--history $(HISTORY) \
		--trials $(TRIALS) \
		--seed $(SEED) \
		--tail-fraction $(TAIL_FRACTION)

lab:
	$(UV_RUN) jupyter lab
