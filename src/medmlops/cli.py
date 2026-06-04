"""MedMLOps-Lab command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from medmlops import __version__
from medmlops.config import load_settings
from medmlops.data.ingest import DEFAULT_OUTPUT_PATH, ingest_diabetes130
from medmlops.data.validate import (
    DEFAULT_INPUT_PATH,
    DEFAULT_METRICS_PATH,
    validate_diabetes130,
)
from medmlops.data.validate import (
    DEFAULT_OUTPUT_PATH as DEFAULT_VALIDATED_PATH,
)
from medmlops.features.pipeline import (
    DEFAULT_FEATURES_PATH,
    DEFAULT_SPLIT_DIR,
    DEFAULT_SPLIT_INPUT_PATH,
    preprocess_features,
    write_splits,
)
from medmlops.logging import configure_logging
from medmlops.seeds import set_deterministic

app = typer.Typer(
    help="Reproducible, CPU-only clinical risk-prediction MLOps commands.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Path to the YAML application config."),
    ] = Path("configs/config.yaml"),
) -> None:
    """Load common settings for all commands."""

    settings = load_settings(config_path)
    configure_logging(settings.logging.level, settings.logging.json_output)
    set_deterministic(settings.project.seed)


@app.command()
def ingest(
    output_path: Annotated[
        Path,
        typer.Option(
            "--output-path",
            "-o",
            help="Where to write the raw Diabetes 130 parquet file.",
        ),
    ] = DEFAULT_OUTPUT_PATH,
) -> None:
    """Download Diabetes 130-US Hospitals through the UCI repository API."""

    destination = ingest_diabetes130(output_path)
    console.print(f"Wrote raw dataset: {destination}")


@app.command()
def validate(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input-path",
            "-i",
            help="Raw Diabetes 130 parquet or CSV input.",
        ),
    ] = DEFAULT_INPUT_PATH,
    output_path: Annotated[
        Path,
        typer.Option("--output-path", "-o", help="Validated parquet output path."),
    ] = DEFAULT_VALIDATED_PATH,
    metrics_path: Annotated[
        Path,
        typer.Option("--metrics-path", "-m", help="DVC metrics JSON output path."),
    ] = DEFAULT_METRICS_PATH,
    check_statistics: Annotated[
        bool,
        typer.Option(
            "--check-statistics/--skip-statistics",
            help="Validate the expected 30-day readmission-rate sanity band.",
        ),
    ] = True,
) -> None:
    """Validate raw data with Pandera contracts."""

    destination = validate_diabetes130(
        input_path=input_path,
        output_path=output_path,
        metrics_path=metrics_path,
        check_statistics=check_statistics,
    )
    console.print(f"Wrote validated dataset: {destination}")


@app.command()
def preprocess(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input-path",
            "-i",
            help="Validated Diabetes 130 parquet input.",
        ),
    ] = DEFAULT_VALIDATED_PATH,
    output_path: Annotated[
        Path,
        typer.Option("--output-path", "-o", help="Feature source parquet output."),
    ] = DEFAULT_FEATURES_PATH,
    params_path: Annotated[
        Path,
        typer.Option("--params", help="DVC params YAML path."),
    ] = Path("params.yaml"),
) -> None:
    """Build the Phase 2 feature source table."""

    destination = preprocess_features(input_path, output_path, params_path)
    console.print(f"Wrote feature source table: {destination}")


@app.command()
def split(
    input_path: Annotated[
        Path,
        typer.Option("--input-path", "-i", help="Feature source parquet input."),
    ] = DEFAULT_SPLIT_INPUT_PATH,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for split parquet files."),
    ] = DEFAULT_SPLIT_DIR,
    params_path: Annotated[
        Path,
        typer.Option("--params", help="DVC params YAML path."),
    ] = Path("params.yaml"),
) -> None:
    """Create deterministic train/validation/calibration/conformal/test splits."""

    paths = write_splits(input_path, output_dir, params_path)
    for name, path in paths.items():
        console.print(f"Wrote {name} split: {path}")


@app.command()
def train(
    train_path: Annotated[
        Path,
        typer.Option("--train-path", help="Training split parquet path."),
    ] = Path("data/processed/train.parquet"),
    val_path: Annotated[
        Path,
        typer.Option("--val-path", help="Validation split parquet path."),
    ] = Path("data/processed/val.parquet"),
    test_path: Annotated[
        Path,
        typer.Option("--test-path", help="Test split parquet path."),
    ] = Path("data/processed/test.parquet"),
    models_dir: Annotated[
        Path,
        typer.Option("--models-dir", help="Directory for trained model artifacts."),
    ] = Path("models"),
    report_path: Annotated[
        Path,
        typer.Option("--report-path", help="Training metrics JSON path."),
    ] = Path("reports/train_metrics.json"),
    params_path: Annotated[
        Path,
        typer.Option("--params", help="DVC params YAML path."),
    ] = Path("params.yaml"),
) -> None:
    """Train the XGBoost baseline and PyTorch tabular MLP hero."""

    from medmlops.models.train import train_phase2

    report = train_phase2(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        models_dir=models_dir,
        report_path=report_path,
        params_path=params_path,
    )
    console.print(f"Wrote training metrics: {report}")


@app.command()
def calibrate(
    hero_path: Annotated[
        Path,
        typer.Option("--hero-path", help="Trained hero model artifact."),
    ] = Path("models/hero.pt"),
    calib_path: Annotated[
        Path,
        typer.Option("--calib-path", help="Calibration split parquet path."),
    ] = Path("data/processed/calib.parquet"),
    val_path: Annotated[
        Path,
        typer.Option("--val-path", help="Validation split parquet path."),
    ] = Path("data/processed/val.parquet"),
    model_path: Annotated[
        Path,
        typer.Option("--model-path", help="Calibrated model output path."),
    ] = Path("models/calibrated.pkl"),
    metrics_path: Annotated[
        Path,
        typer.Option("--metrics-path", help="Calibration metrics JSON path."),
    ] = Path("reports/calibration_metrics.json"),
    figure_path: Annotated[
        Path,
        typer.Option("--figure-path", help="Reliability diagram output path."),
    ] = Path("reports/figures/reliability.png"),
    params_path: Annotated[
        Path,
        typer.Option("--params", help="DVC params YAML path."),
    ] = Path("params.yaml"),
) -> None:
    """Fit probability calibration for the hero model."""

    from medmlops.calibration.calibrate import calibrate_phase3

    metrics = calibrate_phase3(
        hero_path=hero_path,
        calib_path=calib_path,
        val_path=val_path,
        model_path=model_path,
        metrics_path=metrics_path,
        figure_path=figure_path,
        params_path=params_path,
    )
    console.print(f"Wrote calibration metrics: {metrics}")


@app.command()
def conformalize(
    calibrated_path: Annotated[
        Path,
        typer.Option("--calibrated-path", help="Calibrated model artifact."),
    ] = Path("models/calibrated.pkl"),
    conformal_path: Annotated[
        Path,
        typer.Option("--conformal-path", help="Conformal split parquet path."),
    ] = Path("data/processed/conformal.parquet"),
    test_path: Annotated[
        Path,
        typer.Option("--test-path", help="Test split parquet path."),
    ] = Path("data/processed/test.parquet"),
    model_path: Annotated[
        Path,
        typer.Option("--model-path", help="Conformal gate output path."),
    ] = Path("models/conformal.pkl"),
    metrics_path: Annotated[
        Path,
        typer.Option("--metrics-path", help="Conformal metrics JSON path."),
    ] = Path("reports/conformal_metrics.json"),
    params_path: Annotated[
        Path,
        typer.Option("--params", help="DVC params YAML path."),
    ] = Path("params.yaml"),
) -> None:
    """Fit conformal prediction sets and the abstention gate."""

    from medmlops.conformal.conformal import conformalize_phase3

    metrics = conformalize_phase3(
        calibrated_path=calibrated_path,
        conformal_path=conformal_path,
        test_path=test_path,
        model_path=model_path,
        metrics_path=metrics_path,
        params_path=params_path,
    )
    console.print(f"Wrote conformal metrics: {metrics}")


@app.command()
def evaluate(
    conformal_model_path: Annotated[
        Path,
        typer.Option("--conformal-model-path", help="Conformal gate artifact."),
    ] = Path("models/conformal.pkl"),
    test_path: Annotated[
        Path,
        typer.Option("--test-path", help="Test split parquet path."),
    ] = Path("data/processed/test.parquet"),
    metrics_path: Annotated[
        Path,
        typer.Option("--metrics-path", help="Clinical metrics JSON path."),
    ] = Path("reports/clinical_metrics.json"),
    figure_path: Annotated[
        Path,
        typer.Option("--figure-path", help="Decision curve output path."),
    ] = Path("reports/figures/decision_curve.png"),
    params_path: Annotated[
        Path,
        typer.Option("--params", help="DVC params YAML path."),
    ] = Path("params.yaml"),
) -> None:
    """Evaluate calibrated/conformal predictions with clinical metrics."""

    from medmlops.metrics.clinical import evaluate_phase3

    metrics = evaluate_phase3(
        conformal_model_path=conformal_model_path,
        test_path=test_path,
        metrics_path=metrics_path,
        figure_path=figure_path,
        params_path=params_path,
    )
    console.print(f"Wrote clinical metrics: {metrics}")


@app.command("drift-baseline")
def drift_baseline(
    train_path: Annotated[
        Path,
        typer.Option("--train-path", help="Training split used as drift reference."),
    ] = Path("data/processed/train.parquet"),
    output_path: Annotated[
        Path,
        typer.Option("--output-path", help="Drift reference snapshot output path."),
    ] = Path("reports/drift/reference_snapshot.parquet"),
    params_path: Annotated[
        Path,
        typer.Option("--params", help="DVC params YAML path."),
    ] = Path("params.yaml"),
) -> None:
    """Build the input-distribution reference snapshot."""

    from medmlops.monitoring.drift import build_drift_baseline

    destination = build_drift_baseline(train_path, output_path, params_path)
    console.print(f"Wrote drift reference snapshot: {destination}")


@app.command("simulate-drift")
def simulate_drift(
    input_path: Annotated[
        Path,
        typer.Option("--input-path", help="Feature source used for synthetic shifts."),
    ] = Path("data/processed/train.parquet"),
    report_path: Annotated[
        Path,
        typer.Option("--report-path", help="Synthetic drift report JSON path."),
    ] = Path("reports/drift_simulation.json"),
    params_path: Annotated[
        Path,
        typer.Option("--params", help="DVC params YAML path."),
    ] = Path("params.yaml"),
) -> None:
    """Run the clearly labeled synthetic three-regime drift demo."""

    import os

    import pandas as pd

    from medmlops.features.pipeline import load_yaml
    from medmlops.monitoring.simulate import run_synthetic_drift_demo

    params = load_yaml(params_path)
    monitoring = params.get("monitoring", {})
    serving = params.get("serving", {})
    if not isinstance(monitoring, dict) or not isinstance(serving, dict):
        msg = "params.yaml monitoring and serving sections must be mappings"
        raise TypeError(msg)
    database_url = os.environ.get("MEDMLOPS_AUDIT_DATABASE_URL") or str(
        serving.get("audit_database_url", "sqlite:///audit.db")
    )
    destination = run_synthetic_drift_demo(
        pd.read_parquet(input_path),
        feature=str(monitoring.get("drift_feature", "num_medications")),
        database_url=database_url,
        report_path=report_path,
        workspace_path=str(
            monitoring.get("workspace_path", "reports/evidently-workspace")
        ),
        psi_warn=float(monitoring.get("psi_warn", 0.10)),
        psi_alert=float(monitoring.get("psi_alert", 0.25)),
    )
    console.print(f"Wrote SYNTHETIC drift simulation report: {destination}")


@app.command("monitor-performance")
def monitor_performance(
    report_path: Annotated[
        Path,
        typer.Option("--report-path", help="Delayed-label performance report path."),
    ] = Path("reports/performance_monitoring.json"),
    params_path: Annotated[
        Path,
        typer.Option("--params", help="DVC params YAML path."),
    ] = Path("params.yaml"),
) -> None:
    """Run the scheduled delayed-label performance monitoring job."""

    import os

    from medmlops.features.pipeline import load_yaml
    from medmlops.monitoring.performance import monitor_delayed_labels

    params = load_yaml(params_path)
    monitoring = params.get("monitoring", {})
    serving = params.get("serving", {})
    if not isinstance(monitoring, dict) or not isinstance(serving, dict):
        msg = "params.yaml monitoring and serving sections must be mappings"
        raise TypeError(msg)
    database_url = os.environ.get("MEDMLOPS_AUDIT_DATABASE_URL") or str(
        serving.get("audit_database_url", "sqlite:///audit.db")
    )
    destination = monitor_delayed_labels(
        database_url,
        auroc_floor=float(monitoring.get("performance_auroc_floor", 0.60)),
        clinical_threshold=float(monitoring.get("clinical_threshold", 0.10)),
        report_path=report_path,
        workspace_path=str(
            monitoring.get("workspace_path", "reports/evidently-workspace")
        ),
    )
    console.print(f"Wrote delayed-label performance report: {destination}")


@app.command("fairness-audit")
def fairness_audit(
    model_path: Annotated[
        Path,
        typer.Option("--model-path", help="Conformal model artifact."),
    ] = Path("models/conformal.pkl"),
    test_path: Annotated[
        Path,
        typer.Option("--test-path", help="Held-out test split parquet path."),
    ] = Path("data/processed/test.parquet"),
    report_path: Annotated[
        Path,
        typer.Option("--report-path", help="Fairness audit JSON path."),
    ] = Path("reports/fairness.json"),
    figure_path: Annotated[
        Path,
        typer.Option("--figure-path", help="Subgroup AUROC figure path."),
    ] = Path("reports/figures/subgroup_auroc.png"),
    params_path: Annotated[
        Path,
        typer.Option("--params", help="DVC params YAML path."),
    ] = Path("params.yaml"),
) -> None:
    """Audit protected attributes without using them as predictors."""

    from medmlops.fairness.audit import fairness_audit_phase6

    destination = fairness_audit_phase6(
        model_path=model_path,
        test_path=test_path,
        report_path=report_path,
        figure_path=figure_path,
        params_path=params_path,
    )
    console.print(f"Wrote fairness audit: {destination}")


@app.command()
def governance(
    report_dir: Annotated[
        Path,
        typer.Option("--report-dir", help="Pipeline evidence report directory."),
    ] = Path("reports"),
    docs_dir: Annotated[
        Path,
        typer.Option("--docs-dir", help="Generated governance document directory."),
    ] = Path("docs"),
) -> None:
    """Generate evidence-linked responsible-AI governance documents."""

    from medmlops.governance.generate import generate_governance_docs

    for name, destination in generate_governance_docs(report_dir, docs_dir).items():
        console.print(f"Wrote {name}: {destination}")


@app.command()
def version() -> None:
    """Print package version."""

    console.print(__version__)


if __name__ == "__main__":
    app()
