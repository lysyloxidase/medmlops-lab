"""Generate the Diabetes 130 data datasheet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from medmlops.governance.common import nested, write_markdown


def render_datasheet(evidence: dict[str, dict[str, Any]]) -> str:
    """Render a compact Datasheets-for-Datasets style document."""

    quality = evidence["data_quality"]
    train = evidence["train_metrics"]
    return f"""# Datasheet: Diabetes 130-US Hospitals

> **Research and portfolio demonstration only.** This datasheet is an
> aspirational governance artifact, not a certification of dataset fitness.

## Motivation

The dataset supports reproducible study of 30-day readmission-risk modeling and
the operational safeguards around such models.

## Composition

- Source: UCI Machine Learning Repository, Diabetes 130-US Hospitals (1999-2008).
- License: CC BY 4.0.
- Rows validated: `{nested(quality, "row_count")}`
- Positive rate: `{nested(quality, "positive_rate_30d")}`
- Unit of observation: hospital encounter; repeated patients may exist.
- Target: whether readmission occurred within 30 days.

## Collection And Provenance

The source contains encounter records from 130 US hospitals and integrated
delivery networks from 1999-2008. Source hashes and validation evidence are
tracked in DVC and `reports/data_quality.json`.

- Validated source hash: `{nested(quality, "source_sha256")}`
- Training source hash: `{nested(train, "source_train_sha256")}`

## Preprocessing

Diagnosis codes are grouped, numeric values are median-imputed and standardized,
and categorical values are imputed and one-hot encoded. Encounter and patient
identifiers are removed. Race, gender, and age are retained only for fairness
auditing and excluded from model inputs.

## Uses

Appropriate uses are education, reproducibility demonstrations, and research
prototyping. It is not appropriate for direct care, resource allocation,
autonomous triage, or claims about current populations without new validation.

## Risks And Ethical Considerations

Race is a social construct and must not be used as a biological correction.
Historical care patterns can encode inequity. Missingness, coding practices,
repeated encounters, label definition, and temporal age all limit inference.
Subgroup audits are necessary but cannot prove fairness.

## Maintenance

The static source is versioned through DVC. Any replacement or refresh requires
new validation, calibration, fairness auditing, drift baselines, and governance
regeneration.
"""


def generate_datasheet(
    evidence: dict[str, dict[str, Any]],
    output_path: str | Path = "docs/datasheet.md",
) -> Path:
    """Write the dataset datasheet."""

    return write_markdown(output_path, render_datasheet(evidence))
