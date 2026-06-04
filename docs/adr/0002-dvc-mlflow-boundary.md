# ADR 0002: DVC And MLflow Boundary

## Status

Accepted.

## Decision

DVC owns data, pipeline dependencies, generated datasets, and reproducibility
state. MLflow, introduced later, owns experiment metadata, model artifacts, and
run comparison.

## Consequences

Phase 1 does not require MLflow to download or validate data. The boundary keeps
data lineage reproducible even before model training exists.
