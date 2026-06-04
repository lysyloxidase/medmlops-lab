# Architecture

MedMLOps-Lab is organized as six production layers:

1. Data ingestion and versioning with DVC
2. Data contracts and feature boundaries
3. Model training, calibration, conformal intervals, and metrics
4. CPU-only serving
5. Monitoring and data-quality drift checks
6. Fairness and governance documentation

Phase 1 implements the first layer and the contract boundary for the second.
Phase 2 adds sklearn feature engineering, deterministic splits, an XGBoost
baseline, a PyTorch tabular MLP hero model, and MLflow provenance tracking.
