# CI/CD And Release Gates

## Pull Requests And Pushes

The `CI` workflow runs four independent jobs:

1. Ruff lint/format, strict Pyright, and pytest with at least 85% coverage.
2. DVC reproduction with a GitHub Actions-backed local DVC cache and a CML pull
   request comment containing metrics and plots.
3. Performance, calibration, and fairness-regression quality gates.
4. Docker build, full Compose stack startup, and live API integration tests.

The performance gate requires held-out AUROC of at least `0.62`. The calibration
gate requires ECE no greater than `0.03`.

The fairness gate prevents subgroup AUROC gaps from widening more than `0.10`
from the committed baseline. It is deliberately a regression gate rather than a
claim that current observed gaps are acceptable. Existing gaps remain visible
in the Model Card and fairness report.

## Independent Reproduction

Releases and manual dispatches build the Docker image without cache and
reproduce all metric reports from scratch. Canonical hashes are compared with
`reports/reference_hashes.json`; any divergence fails the job.

## Release

A `v*` tag first calls the independent-reproduction workflow. Only after that
gate passes does the release workflow build and publish:

- `ghcr.io/lysyloxidase/medmlops-lab:<tag>`
- `ghcr.io/lysyloxidase/medmlops-lab:latest`
- the MkDocs site on `gh-pages`

The image is a reproducible platform image. Model and data artifacts remain DVC
outputs and are not embedded as opaque release assets.

## Local Hooks

`make setup` installs pre-commit, commit-message, and pre-push hooks. They
enforce Ruff, formatting, strict Pyright, notebook stripping, Conventional
Commits, and the unit-test coverage floor.
