# ADR 0003: Race As Audit Only

## Status

Accepted.

## Decision

Race, gender, and age are configured as audit-only fields by default. They are
validated as raw columns but excluded from default predictor sets.

## Consequences

The platform can report performance and calibration differences across groups
without silently using sensitive demographic fields as predictors.

Race is treated as a social and political construct, not a biological
correction factor. The removal of race coefficients from eGFR equations is a
cautionary example: embedding race can disguise structural inequity as biology.
Accordingly, this project uses race only to identify possible disparate
performance and never to adjust an individual's risk.

Fairness definitions can also conflict. When groups have unequal base rates,
calibration and equalized odds generally cannot both be achieved except in
special cases. The Phase 6 audit reports this trade-off rather than claiming a
single metric establishes fairness.
