"""Generate a 27-item TRIPOD+AI evidence checklist."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from medmlops.governance.common import write_markdown

TRIPOD_AI_ITEMS: Final[tuple[tuple[str, str], ...]] = (
    ("Title identifies prediction-model study", "README.md"),
    ("Structured summary", "README.md; docs/model-card.md"),
    ("Background and rationale", "README.md; docs/model-card.md"),
    ("Objectives", "README.md"),
    ("Source of data", "docs/datasheet.md"),
    ("Study setting", "docs/datasheet.md"),
    ("Eligibility criteria", "docs/datasheet.md"),
    ("Outcome definition", "params.yaml; docs/datasheet.md"),
    ("Candidate predictors", "params.yaml"),
    ("Sample size", "reports/data_quality.json"),
    ("Missing data handling", "src/medmlops/features/pipeline.py"),
    ("Data preparation", "src/medmlops/features/pipeline.py"),
    ("Model type and specification", "src/medmlops/models/"),
    ("Model-building procedures", "src/medmlops/models/train.py"),
    ("Internal validation", "dvc.yaml; reports/train_metrics.json"),
    ("Performance measures", "reports/clinical_metrics.json"),
    ("Fairness assessment", "reports/fairness.json"),
    ("Model output and thresholds", "params.yaml; reports/clinical_metrics.json"),
    ("Participant flow", "dvc.yaml; reports/data_quality.json"),
    ("Participant characteristics", "docs/datasheet.md"),
    ("Model performance", "reports/clinical_metrics.json"),
    ("Model specification availability", "models/; src/medmlops/models/"),
    ("Interpretation", "docs/model-card.md"),
    ("Limitations", "docs/caveats.md; docs/model-card.md"),
    ("Implications and future research", "docs/model-card.md"),
    ("Supplementary information and protocol", "docs/; dvc.yaml"),
    ("Funding, conflicts, registration, and data/code access", "README.md; LICENSE"),
)


def render_tripod_ai() -> str:
    """Render all 27 checklist items with evidence pointers."""

    rows = "\n".join(
        f"| {index} | {item} | Addressed | `{evidence}` |"
        for index, (item, evidence) in enumerate(TRIPOD_AI_ITEMS, start=1)
    )
    return f"""# TRIPOD+AI Self-Assessment

> This is an aspirational, project-level self-assessment against the 27-item
> TRIPOD+AI reporting checklist. It is not independent review or certification.

TRIPOD+AI supersedes the original TRIPOD statement for clinical prediction
models using regression or machine-learning methods. Evidence pointers identify
where this repository addresses each item; "Addressed" does not imply adequacy.

| # | Reporting item | Status | Evidence pointer |
|---:|---|---|---|
{rows}

Source: Collins GS et al., *BMJ* 2024;385:e078378,
<https://www.bmj.com/content/385/bmj-2023-078378>.
"""


def generate_tripod_ai(output_path: str | Path = "docs/tripod-ai.md") -> Path:
    """Write the 27-item checklist."""

    return write_markdown(output_path, render_tripod_ai())
