# Independent Reproduction

Reproducibility is a release gate, not a prose-only aspiration.

```bash
make reproduce
```

From the host, this command builds and runs the pinned ARM64 reference
container. Inside that container it forces every DVC stage, then canonicalizes
and SHA-256 hashes:

- `reports/train_metrics.json`
- `reports/calibration_metrics.json`
- `reports/conformal_metrics.json`
- `reports/clinical_metrics.json`
- `reports/fairness.json`

The command fails if any produced hash differs from
`reports/reference_hashes.json`. JSON keys and whitespace are canonicalized;
metric content is not rounded or tolerated by the hash gate. The manifest
declares one deliberately excluded non-metric field:
`fairness.json.provenance.model_sha256`. Joblib artifact bytes vary between
otherwise identical runs; the provenance remains in the report, while every
fairness value and the stable test-data digest remain exact-hashed.

`make reproduce-local` runs the same checks directly in the current
environment for investigation, but only the pinned ARM64 container is the
release reference.

## Independent-Container Gate

The `Independent Reproduction` GitHub Actions workflow builds
`docker/Dockerfile` with `--no-cache`, runs the entire pipeline in a fresh
container, and verifies the output against the committed reference manifest.
This operationalizes the fully automated analysis principle associated with the
gold reproducibility standard described by Heil et al. (2021).

Reference:
[Heil et al., Nature Methods 2021](https://www.nature.com/articles/s41592-021-01256-7).

## Inputs And Controls

- `uv.lock` pins Python dependency resolution.
- The Dockerfile pins the uv build image and Python minor version.
- `params.yaml` is the experiment parameter source of truth.
- `dvc.yaml` and `dvc.lock` define pipeline state and dependencies.
- Raw, split, and model artifacts carry SHA-256 or DVC hashes.
- A centralized seed configures Python, NumPy, and PyTorch.
- CPU execution is single-threaded where deterministic behavior matters.
- Metric reports contain no timestamps or MLflow run IDs.

## Honest Scope

The exact-hash guarantee is scoped to the pinned Docker image on the reference
ARM64 CPU architecture. PyTorch explicitly warns that completely reproducible results
are not guaranteed across releases, platforms, or CPU/GPU execution. A
cross-architecture divergence is a failed exact reproduction, but not by itself
proof of a clinically meaningful regression.

Reference:
[PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Local macOS runs may use the deterministic sklearn GBDT fallback when the native
OpenMP runtime required by XGBoost is absent. Those runs remain useful for
development but are not the release reference environment.
