.PHONY: setup data train clinical monitor responsible-ai serve evidently-ui reproduce reproduce-local reference-hashes test lint docs format clean

setup: ## Install frozen dependencies and all pre-commit hook types.
	uv sync --frozen --all-groups
	@if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then \
		uv run pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push; \
	else \
		echo "Skipping pre-commit install; not inside a git repository."; \
	fi

data: ## Download, validate, preprocess, and split data.
	uv run dvc repro ingest validate preprocess split

train: ## Run the training stage and its dependencies.
	uv run dvc repro train

clinical: ## Run calibration, conformalization, and clinical evaluation.
	uv run dvc repro calibrate conformalize evaluate

monitor: ## Build drift baseline and run monitoring demonstrations.
	uv run dvc repro drift_baseline
	uv run medmlops simulate-drift
	uv run medmlops monitor-performance

responsible-ai: ## Run fairness audit and generate governance documents.
	uv run dvc repro fairness_audit governance

serve: ## Launch the API, PostgreSQL, MLflow, and MinIO stack.
	uv run dvc repro
	docker compose -f docker/docker-compose.yml up

evidently-ui: ## Launch the local Evidently workspace UI.
	uv run evidently ui --workspace reports/evidently-workspace --port 8001

reproduce: ## Reproduce and verify metrics in the pinned ARM64 Docker environment.
	@if [ -f /.dockerenv ]; then \
		make reproduce-local; \
	else \
		docker build --platform linux/arm64 -f docker/Dockerfile -t medmlops:repro .; \
		docker run --rm --platform linux/arm64 medmlops:repro make reproduce-local; \
	fi

reproduce-local: ## Run the reproduction gate in the current environment.
	uv run dvc repro --force
	uv run medmlops verify-reproduction \
		--produced reports/ \
		--reference reports/reference_hashes.json \
		--require-architecture

reference-hashes: ## Write hashes only after a trusted reference reproduction.
	uv run medmlops write-reference-hashes \
		--reports reports/ \
		--output reports/reference_hashes.json

test: ## Run lint, strict types, and tests with the release coverage floor.
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	uv run pytest --cov=medmlops --cov-fail-under=85

lint: ## Run Ruff and strict Pyright.
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright

docs: ## Serve the MkDocs site locally.
	uv run mkdocs serve

format: ## Format and apply safe Ruff fixes.
	uv run ruff format .
	uv run ruff check --fix .

clean: ## Remove local test and documentation build outputs.
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov site
