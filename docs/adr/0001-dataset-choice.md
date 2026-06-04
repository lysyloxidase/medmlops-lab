# ADR 0001: Dataset Choice

## Status

Accepted.

## Context

The platform needs a public clinical tabular dataset that can be redistributed
and reproduced without private data access.

## Decision

Use the Diabetes 130-US Hospitals dataset from the UCI Machine Learning
Repository, ID 296.

## Consequences

The dataset is large enough to exercise production data contracts and class
imbalance, but it has no clean event timestamp for temporal splitting. Race,
gender, and age are configured as audit-only fields by default.
