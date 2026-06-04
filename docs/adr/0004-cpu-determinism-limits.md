# ADR 0004: CPU Determinism Limits

## Status

Accepted.

## Context

Numerical software stacks can change results across library releases, platforms,
instruction sets, thread scheduling, and hardware.

## Decision

MedMLOps-Lab claims CPU reproducibility only within the pinned Docker image on a
given CPU architecture. The seed utility sets Python, NumPy, BLAS threading, and
PyTorch deterministic defaults.

## Consequences

Documentation and model cards must avoid claiming universal reproducibility
across PyTorch releases, commits, platforms, or CPU/GPU boundaries.
