# ADR 0003: Race As Audit Only

## Status

Accepted.

## Decision

Race, gender, and age are configured as audit-only fields by default. They are
validated as raw columns but excluded from default predictor sets.

## Consequences

The platform can report performance and calibration differences across groups
without silently using sensitive demographic fields as predictors.
