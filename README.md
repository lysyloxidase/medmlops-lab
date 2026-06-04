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
```

The `make data` target runs:

```bash
uv run dvc repro ingest validate preprocess split
```

The `make train` target runs:

```bash
uv run dvc repro train
```

Raw data is written to `data/raw/diabetes130.parquet` and remains outside git.
The validated Phase 1 artifact is written to `data/interim/validated.parquet`,
with summary metrics in `reports/data_quality.json`. Phase 2 writes split
feature tables to `data/processed/`, model artifacts to `models/`, and stable
training metrics to `reports/train_metrics.json`.

## Phase 2 Training

Every training run fits both models:

- XGBoost baseline, the mandatory tabular-data reference model
- PyTorch tabular MLP hero model, trained CPU-only with class-weighted BCE

The README and MLflow runs report both scores. If the MLP does not match or beat
the baseline, the baseline is registered as `models:/MedMLOps@champion`.

The baseline prefers XGBoost. On macOS hosts without `libomp.dylib`, the local
training code falls back to sklearn's histogram GBDT so the pipeline still
reproduces; the Docker image installs the Linux OpenMP runtime for XGBoost.

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
make serve       # Phase 4 placeholder
```
