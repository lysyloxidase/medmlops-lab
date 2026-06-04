"""Backup NHANES adapter placeholder.

Phase 1 establishes the adapter boundary but intentionally keeps Diabetes 130 as
the only default dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class NHANESAdapterConfig:
    """Configuration for a future NHANES adapter."""

    source_dir: Path
    output_path: Path


class NHANESAdapter:
    """Minimal adapter boundary for future NHANES support."""

    def __init__(self, config: NHANESAdapterConfig) -> None:
        self.config = config

    def transform(self) -> pd.DataFrame:
        """Transform NHANES files into the platform contract.

        The implementation is deferred until a specific NHANES cohort and task
        definition are selected.
        """

        msg = "NHANES adapter implementation is deferred beyond Phase 1"
        raise NotImplementedError(msg)
