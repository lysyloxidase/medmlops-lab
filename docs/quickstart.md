# Quickstart

Install `uv` first:

https://docs.astral.sh/uv/getting-started/installation/

```bash
make setup
make test
make data
make train
make clinical
```

`make data` runs the DVC stages:

```bash
uv run dvc repro ingest validate preprocess split
```

Outputs:

- `data/raw/diabetes130.parquet`
- `data/raw/diabetes130.sha256`
- `data/interim/validated.parquet`
- `data/processed/features.parquet`
- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/calib.parquet`
- `data/processed/conformal.parquet`
- `data/processed/test.parquet`
- `reports/data_quality.json`
- `models/hero.pt`
- `models/baseline.pkl`
- `reports/train_metrics.json`
- `models/calibrated.pkl`
- `models/conformal.pkl`
- `reports/calibration_metrics.json`
- `reports/conformal_metrics.json`
- `reports/clinical_metrics.json`
- `reports/figures/reliability.png`
- `reports/figures/decision_curve.png`
