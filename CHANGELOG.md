# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-06-04

### Added

- Phase 1 repository scaffold.
- DVC ingest and validation stages for the Diabetes 130-US Hospitals dataset.
- Pandera data contracts and focused unit tests.
- CPU determinism utility and project configuration.
- Phase 2 sklearn feature engineering, deterministic splits, XGBoost-preferred
  GBDT baseline, PyTorch tabular MLP hero model, MLflow provenance tracking,
  model registry aliasing, and training metrics.
- Phase 3 isotonic/Platt/temperature calibration, reliability diagrams, MAPIE
  conformal abstention, and clinical metrics including decision-curve net
  benefit.
- Phase 4 FastAPI serving with Pydantic v2 clinical validation, MLflow alias
  warm-loading, conformal/OOD abstention, PostgreSQL prediction audit logging,
  Prometheus metrics, model-info, batch prediction, and challenger shadow
  routing.
- Phase 5 Evidently drift reports/test suites, transparent PSI/KS/Wasserstein/MMD
  metrics with persisted alerts, synthetic three-regime drift simulation, and
  retrospective delayed-label performance monitoring.
- Phase 6 Fairlearn subgroup discrimination/calibration audit with audit-only
  protected attributes, a fairness regression gate, and generated Model Card,
  Datasheet, TRIPOD+AI, PROBAST+AI, FDA GMLP, and EU AI Act governance framing.
- Phase 7 GitHub Actions CI/CD with CML pull-request reports, performance /
  calibration / fairness-regression gates, Docker Compose integration tests,
  no-cache independent reproduction with canonical metric hashes, GHCR
  publishing, MkDocs deployment, and v1.0.0 release automation.

[Unreleased]: https://github.com/lysyloxidase/medmlops-lab/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/lysyloxidase/medmlops-lab/releases/tag/v1.0.0
