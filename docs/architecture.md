# Architecture

MedMLOps-Lab v1.0.0 is organized as seven production-shaped layers:

1. **Data and contracts**: UCI ingestion, DVC versioning, Pandera validation,
   deterministic splits, and leakage boundaries.
2. **Model development**: XGBoost-preferred baseline, PyTorch tabular MLP, CPU
   determinism, and MLflow provenance.
3. **Clinical rigor**: probability calibration, MAPIE conformal prediction,
   abstention, AUPRC, calibration metrics, and decision-curve analysis.
4. **Serving**: FastAPI, Pydantic validation, OOD checks, alias-based registry
   loading, PostgreSQL prediction logging, and Prometheus metrics.
5. **Monitoring**: Evidently, transparent PSI/KS/Wasserstein/MMD, synthetic
   drift regimes, and delayed-label performance monitoring.
6. **Responsible AI and governance**: audit-only protected attributes,
   Fairlearn subgroup audits, Model Card, Datasheet, TRIPOD+AI, PROBAST+AI, and
   cautious regulatory framing.
7. **Release engineering**: GitHub Actions, CML, quality gates, Docker
   integration, canonical metric hashes, no-cache independent reproduction,
   GHCR publishing, and MkDocs deployment.

The primary ownership boundary is deliberate: race, gender, and age stay
available for auditing but never enter the predictor matrix.
