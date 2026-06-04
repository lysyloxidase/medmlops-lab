.PHONY: setup data train clinical serve reproduce test lint format clean

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

serve:
	@echo "Serving is introduced in Phase 4."

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
