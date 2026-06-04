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
def version() -> None:
    """Print package version."""

    console.print(__version__)


if __name__ == "__main__":
    app()
