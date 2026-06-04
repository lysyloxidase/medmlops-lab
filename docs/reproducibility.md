# Reproducibility

The project treats reproducibility as a user-facing feature. Phase 1 uses:

- `uv.lock` for Python dependency resolution
- `params.yaml` as a single source of truth for experiment parameters
- `dvc.yaml` and `dvc.lock` for pipeline state
- SHA-256 logging for raw downloaded data
- a centralized seed function in `src/medmlops/seeds.py`
- deterministic sklearn splits and single-threaded CPU model training
- stable `reports/train_metrics.json` output with no timestamps or run IDs

The honest guarantee is scoped: bit-reproducible runs are expected only inside a
pinned Docker image on the same CPU architecture.

On macOS without the native OpenMP runtime required by XGBoost, local runs use a
deterministic sklearn histogram-GBDT fallback. The Docker image installs
`libgomp1` so Linux container runs can use XGBoost.
