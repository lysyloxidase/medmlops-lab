# Monitoring

Phase 5 combines Evidently reports with transparent custom drift metrics.

## Drift Baseline

The DVC `drift_baseline` stage writes the training-distribution reference to
`reports/drift/reference_snapshot.parquet`.

```bash
uv run dvc repro drift_baseline
```

## Synthetic Demo

The three-regime simulation injects covariate shift `P(X)`, prior probability
shift `P(Y)`, and concept shift `P(Y|X)` into the static Diabetes 130 data.

```bash
uv run medmlops simulate-drift
```

Every simulation report is labeled **SYNTHETIC DRIFT DEMONSTRATION**. These
shifts are injected for an engineering demonstration and are not evidence of
observed real-world drift. The command logs all three labeled snapshots into the
local Evidently workspace for inspection in the UI.

## Delayed Labels

Run the retrospective performance job after ground truth has been attached to
rows in the `predictions` audit table:

```bash
uv run medmlops monitor-performance
```

Schedule this command nightly with the deployment scheduler. It recomputes
AUROC, AUPRC, calibration, and net benefit only on the currently labeled
subset, and raises an alert when AUROC is below the configured floor.

The repository includes
`.github/workflows/nightly-performance-monitoring.yml`, scheduled for 02:00 UTC.
Set the `MEDMLOPS_AUDIT_DATABASE_URL` repository secret to point it at the
production-like PostgreSQL prediction log.

## Evidently UI

Launch the self-hosted workspace UI:

```bash
make evidently-ui
```

The UI is served at `http://localhost:8001`, avoiding the Phase 4 FastAPI port
at `http://localhost:8000`.
