.PHONY: setup data train clinical monitor responsible-ai serve evidently-ui reproduce test lint format clean

setup:
	uv sync --all-groups
	@if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then \
		uv run pre-commit install; \
	else \
		echo "Skipping pre-commit install; not inside a git repository."; \
	fi

data:
	uv run dvc repro ingest validate preprocess split

train:
	uv run dvc repro train

clinical:
	uv run dvc repro calibrate conformalize evaluate

monitor:
	uv run dvc repro drift_baseline
	uv run medmlops simulate-drift
	uv run medmlops monitor-performance

responsible-ai:
	uv run dvc repro fairness_audit governance

serve:
	uv run uvicorn medmlops.serving.app:app --host 0.0.0.0 --port 8000

evidently-ui:
	uv run evidently ui --workspace reports/evidently-workspace --port 8001

reproduce:
	uv run dvc repro

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run pyright

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov site
