# MedMLOps-Lab

Reproducible, CPU-only clinical risk-prediction engineering for a portfolio
setting. This repository is intentionally built as a production-shaped platform,
not as a notebook.

> Portfolio demo only. MedMLOps-Lab is not a medical device and must not be used
> for patient-care decisions.

## Five-Pillar Thesis

1. **Reproducibility first**: DVC stages, pinned params, and lock files make data
   and model runs auditable.
2. **Contracts before modeling**: Pandera schemas validate clinical tabular data
   before features or training code can consume it.
3. **CPU-only determinism**: the platform targets repeatable runs inside a
   pinned Docker image on a fixed CPU architecture.
4. **Clinical honesty**: class imbalance, leakage columns, missingness, and
   fairness caveats are surfaced rather than hidden.
5. **Governance by default**: docs, templates, licensing, and disclaimers are
   first-class engineering artifacts.

## Quickstart

Prerequisites:

- `uv` installed: https://docs.astral.sh/uv/getting-started/installation/
- Docker, only if you want to run the local Postgres, MLflow, and MinIO stack

```bash
make setup
make test
make data
make train
make clinical
```

The `make data` target runs:

```bash
uv run dvc repro ingest validate preprocess split
```

The `make train` target runs:

```bash
uv run dvc repro train
```

The `make clinical` target runs:

```bash
uv run dvc repro calibrate conformalize evaluate
```

Raw data is written to `data/raw/diabetes130.parquet` and remains outside git.
The validated Phase 1 artifact is written to `data/interim/validated.parquet`,
with summary metrics in `reports/data_quality.json`. Phase 2 writes split
feature tables to `data/processed/`, model artifacts to `models/`, and stable
training metrics to `reports/train_metrics.json`.

Phase 3 writes calibrated and conformal artifacts to `models/`, clinical metrics
to `reports/clinical_metrics.json`, and reliability/decision-curve plots under
`reports/figures/`.

Phase 4 serves the calibrated conformal gate through FastAPI. The app warm-loads
the MLflow `models:/MedMLOps@champion` alias at startup, applies Pydantic v2
clinical validation, abstains on ambiguous/empty conformal sets or OOD numeric
quantile flags, exposes Prometheus metrics, and writes append-only prediction
audit rows to PostgreSQL in Docker or `audit.db` locally.

Phase 5 builds an Evidently training-distribution baseline, transparent custom
PSI/KS/Wasserstein/MMD checks, a clearly labeled synthetic three-regime drift
demo, and retrospective delayed-label performance monitoring from the prediction
audit log.

## Phase 2 Training

Every training run fits both models:

- XGBoost baseline, the mandatory tabular-data reference model
- PyTorch tabular MLP hero model, trained CPU-only with class-weighted BCE

The README and MLflow runs report both scores. If the MLP does not match or beat
the baseline, the baseline is registered as `models:/MedMLOps@champion`.

The baseline prefers XGBoost. On macOS hosts without `libomp.dylib`, the local
training code falls back to sklearn's histogram GBDT so the pipeline still
reproduces; the Docker image installs the Linux OpenMP runtime for XGBoost.

## Phase 3 Clinical Rigor

The clinical layer fits isotonic, Platt, or temperature calibration on a held-out
calibration split, then wraps the calibrated model in a MAPIE conformal
abstention gate. Ambiguous or empty prediction sets abstain by default.

Reports include ECE, Brier score, calibration slope/intercept, AUPRC,
sensitivity/specificity/PPV/NPV at clinical thresholds, decision-curve net
benefit, and conformal marginal coverage.

## Phase 4 Serving

Run the API locally after the Phase 3 artifacts exist:

```bash
make serve
```

The main endpoints are `POST /predict`, `POST /batch-predict`, `GET /health`,
`GET /metrics`, and `GET /model-info`. Docker Compose binds the app on port
8000 and uses the Compose PostgreSQL service for the append-only `predictions`
audit table.

## Phase 5 Monitoring

```bash
make monitor
make evidently-ui
```

The local Evidently workspace UI runs on `http://localhost:8001`. Synthetic
drift outputs are demonstrations only; delayed-label performance reports are
retrospective and reflect only the labeled subset.

## Dataset

Default source: Diabetes 130-US Hospitals for Years 1999-2008, UCI Machine
Learning Repository ID 296.

- Citation: Strack et al., BioMed Research International 2014
- DOI: `10.24432/C5230J`
- Article DOI: `10.1155/2014/781670`
- License: CC BY 4.0
- Target: 30-day readmission, encoded from `readmitted == "<30"`

## Common Commands

```bash
make setup       # install dependencies with uv
make data        # run DVC ingest, validation, preprocess, and split stages
make reproduce   # reproduce every DVC stage
make test        # run unit tests
make train       # train baseline + hero and log MLflow provenance
make clinical    # calibrate, conformalize, and evaluate clinical metrics
make monitor     # build drift baseline, simulate drift, monitor delayed labels
make serve       # run the FastAPI serving layer on http://localhost:8000
```
