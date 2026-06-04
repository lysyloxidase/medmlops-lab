from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_diabetes130_path() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_diabetes130.csv"


@pytest.fixture
def sample_diabetes130(sample_diabetes130_path: Path) -> pd.DataFrame:
    return pd.read_csv(sample_diabetes130_path, keep_default_na=False)
