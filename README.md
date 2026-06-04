# MedMLOps-Lab

[![CI](https://github.com/lysyloxidase/medmlops-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/lysyloxidase/medmlops-lab/actions/workflows/ci.yml)
[![Independent Reproduction](https://github.com/lysyloxidase/medmlops-lab/actions/workflows/independent-reproduction.yml/badge.svg)](https://github.com/lysyloxidase/medmlops-lab/actions/workflows/independent-reproduction.yml)
[![Docs](https://img.shields.io/badge/docs-MkDocs-526CFE)](https://lysyloxidase.github.io/medmlops-lab/)

A production-shaped, CPU-only clinical risk-prediction platform that is
simultaneously:

1. 🔁 **Metric-hash reproducible** within the pinned Docker image on its
   reference CPU architecture (`make reproduce`)
2. 📋 **TRIPOD+AI mapped** with auto-generated evidence-linked reporting
3. ⚖️ **Continuously fairness-audited**, with race retained for audit only
4. 📉 **Drift-monitored**, including retrospective delayed-label performance
5. 🛑 **Protected by a conformal abstention gate** for ambiguous predictions

This is an engineering and portfolio demonstration, not a notebook.

> **Portfolio and educational demo only. MedMLOps-Lab is not FDA-cleared,
> CE-marked, or a medical device, and must not be used for patient care.**

## Laptop Quickstart

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/)
and Docker.

```bash
git clone https://github.com/lysyloxidase/medmlops-lab
cd medmlops-lab
make setup
make test
make reproduce
make serve
```

`make reproduce` forces every DVC stage and fails if any canonical metric hash
diverges from `reports/reference_hashes.json`. The exact hash claim is scoped to
the pinned Docker image and reference CPU architecture; cross-architecture
equality is not promised.

`make serve` materializes any missing local DVC artifacts, then launches the
API, PostgreSQL prediction log, MLflow, and MinIO with Docker Compose. The API
is available at `http://localhost:8000`.

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

Phase 6 audits race, gender, and age as audit-only attributes with Fairlearn,
including subgroup discrimination and calibration, then generates an
evidence-linked Model Card, Datasheet, TRIPOD+AI checklist, PROBAST+AI
self-assessment, and cautious FDA GMLP / EU AI Act framing.

Phase 7 adds GitHub Actions CI/CD, CML pull-request reports, performance /
calibration / fairness-regression gates, Docker integration testing, GHCR
publishing, MkDocs deployment, and a no-cache independent-reproduction release
gate.

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

## Phase 6 Responsible AI And Governance

```bash
make responsible-ai
```

The fairness audit never adds sensitive attributes to the predictor matrix.
Governance mappings are aspirational self-assessments, not regulatory
certification or evidence of clinical fitness.

## Phase 7 CI/CD And Independent Reproduction

Every push and pull request runs Ruff, strict Pyright, tests with at least 85%
coverage, `dvc repro`, CML reporting, release quality gates, and full-stack
Docker integration tests. Releases additionally build the image without cache,
reproduce the pipeline in a fresh container, and compare canonical metric
hashes against the committed reference.

The fairness gate is a regression gate: it prevents any subgroup AUROC gap from
widening more than `0.10` beyond the committed baseline. It does not conceal or
claim resolution of the existing race and age gaps documented in the Model
Card.

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
make reproduce   # force every DVC stage and verify metric hashes
make test        # lint, type-check, and test with >=85% coverage
make train       # train baseline + hero and log MLflow provenance
make clinical    # calibrate, conformalize, and evaluate clinical metrics
make monitor     # build drift baseline, simulate drift, monitor delayed labels
make responsible-ai # run fairness audit and generate governance documents
make serve       # launch the full Docker Compose stack
make docs        # serve the MkDocs site locally
```
