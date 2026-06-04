"""Orchestrate generated governance artifacts."""

from __future__ import annotations

from pathlib import Path

from medmlops.governance.common import evidence_bundle
from medmlops.governance.datasheet import generate_datasheet
from medmlops.governance.model_card import generate_model_card
from medmlops.governance.probast_ai import generate_probast_ai
from medmlops.governance.regulatory import generate_regulatory_framing
from medmlops.governance.tripod_ai import generate_tripod_ai


def generate_governance_docs(
    report_dir: str | Path = "reports",
    docs_dir: str | Path = "docs",
) -> dict[str, Path]:
    """Generate the Phase 6 governance document set from pipeline evidence."""

    evidence = evidence_bundle(report_dir)
    root = Path(docs_dir)
    return {
        "model_card": generate_model_card(evidence, root / "model-card.md"),
        "datasheet": generate_datasheet(evidence, root / "datasheet.md"),
        "tripod_ai": generate_tripod_ai(root / "tripod-ai.md"),
        "probast_ai": generate_probast_ai(root / "probast-ai.md"),
        "regulatory_framing": generate_regulatory_framing(
            root / "regulatory-framing.md"
        ),
    }
