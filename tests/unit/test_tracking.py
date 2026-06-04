from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import mlflow
import mlflow.sklearn
from sklearn.dummy import DummyClassifier

from medmlops.models.tracking import (
    get_champion_model_version,
    register_champion,
)


def test_mlflow_champion_alias_is_resolvable(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("alias-test")
    model_name = f"MedMLOpsTest-{uuid4().hex}"
    classifier = DummyClassifier(strategy="prior")
    classifier.fit([[0], [1], [2], [3]], [0, 0, 1, 1])

    with mlflow.start_run() as run:
        model_info = mlflow.sklearn.log_model(classifier, name="model")
        run_id = run.info.run_id

    register_champion(
        run_id,
        model_name=model_name,
        model_uri=str(model_info.model_uri),
    )
    champion = get_champion_model_version(model_name)

    assert champion.name == model_name
    assert int(champion.version) == 1


def test_deprecated_mlflow_stage_api_is_not_used() -> None:
    source_root = Path("src")
    source = "\n".join(path.read_text() for path in source_root.rglob("*.py"))

    assert "Staging" not in source
    assert "Production" not in source
