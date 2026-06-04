"""Download Diabetes 130-US Hospitals dataset.

Source: UCI ML Repository ID 296 (Strack, Cios, DeShazo, Clore 2014)
DOI: 10.24432/C5230J
License: CC BY 4.0
Citation: Strack et al., BioMed Research International 2014,
          doi:10.1155/2014/781670
Size: 101,766 encounters x 50 columns including target
Target: 30-day readmission (positive rate roughly 11%)

The download is wrapped as a DVC stage so `dvc repro ingest` is the single
source of truth. Raw data is not committed to git.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

DEFAULT_OUTPUT_PATH = Path("data/raw/diabetes130.parquet")

MISSING_TOKEN_FILL_VALUES = {
    "race": "?",
    "weight": "?",
    "payer_code": "?",
    "medical_specialty": "?",
    "diag_1": "?",
    "diag_2": "?",
    "diag_3": "?",
    "max_glu_serum": "None",
    "A1Cresult": "None",
}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for a file."""

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ingest_diabetes130(output_path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Fetch Diabetes 130 and write it to a parquet file."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    dataset = fetch_ucirepo(id=296)
    data = getattr(dataset, "data", None)
    ids = getattr(data, "ids", None)
    features = getattr(data, "features", None)
    targets = getattr(data, "targets", None)
    if features is None or targets is None:
        msg = "UCI dataset response did not include features and targets"
        raise RuntimeError(msg)

    frames = [frame for frame in (ids, features, targets) if frame is not None]
    df = pd.concat(frames, axis=1)
    df = normalize_missing_tokens(df)
    df.to_parquet(destination, index=False)

    digest = sha256_file(destination)
    digest_path = destination.with_suffix(".sha256")
    digest_path.write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    return destination


def normalize_missing_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Restore documented Diabetes 130 categorical missing-value tokens."""

    output = df.copy()
    for column, fill_value in MISSING_TOKEN_FILL_VALUES.items():
        if column in output.columns:
            output[column] = output[column].fillna(fill_value)
    return output
