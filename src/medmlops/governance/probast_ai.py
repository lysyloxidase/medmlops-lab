"""Generate a PROBAST+AI risk-of-bias self-assessment."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from medmlops.governance.common import write_markdown

PROBAST_AI_DOMAINS: Final[dict[str, tuple[str, ...]]] = {
    "Participants and data sources": (
        "Were data sources appropriate for the intended use?",
        "Were participant inclusion and exclusions appropriate?",
        "Was the data collection setting adequately described?",
        "Are representativeness and transportability limitations addressed?",
    ),
    "Predictors": (
        "Were predictors defined and measured consistently?",
        "Were predictors available at the intended prediction time?",
        "Were protected attributes prevented from unintended model use?",
        "Was predictor missingness handled appropriately?",
    ),
    "Outcome": (
        "Was the outcome determined appropriately?",
        "Was the outcome definition prespecified and consistent?",
        "Was outcome determination independent of predictor knowledge?",
        "Was the prediction horizon clinically meaningful?",
    ),
    "Analysis": (
        "Was the sample size and event count adequate?",
        "Were modeling and validation procedures appropriate?",
        "Were discrimination, calibration, utility, and fairness evaluated?",
        "Were overfitting, missing data, and optimism addressed?",
    ),
}

DOMAIN_RATINGS: Final[dict[str, tuple[str, str]]] = {
    "Participants and data sources": (
        "High concern",
        "Historical, multi-site retrospective data cannot establish current "
        "local transportability.",
    ),
    "Predictors": (
        "Some concern",
        "Predictors are reproducible and audit-only attributes are excluded, "
        "but historical coding can encode inequity.",
    ),
    "Outcome": (
        "Some concern",
        "The 30-day readmission label is reproducible but is an imperfect proxy "
        "for preventable clinical harm.",
    ),
    "Analysis": (
        "Some concern",
        "Internal holdouts, calibration, conformal prediction, fairness, and "
        "monitoring are present; external and prospective validation are absent.",
    ),
}


def render_probast_ai() -> str:
    """Render four domains and sixteen signaling questions."""

    sections: list[str] = []
    question_number = 1
    for domain, questions in PROBAST_AI_DOMAINS.items():
        rating, rationale = DOMAIN_RATINGS[domain]
        rows = "\n".join(
            f"| {question_number + offset} | {question} | Yes / concern noted |"
            for offset, question in enumerate(questions)
        )
        question_number += len(questions)
        sections.append(
            f"""## {domain}

**Risk-of-bias judgment: {rating}.** {rationale}

| # | Signaling question | Self-assessment |
|---:|---|---|
{rows}"""
        )
    return (
        """# PROBAST+AI Self-Assessment

> Aspirational self-assessment only. This is not an independent PROBAST+AI
> appraisal, regulatory determination, or certification.

PROBAST+AI assesses risk of bias and applicability for prediction-model studies.
This repository maps 16 signaling questions across its four domains.

"""
        + "\n\n".join(sections)
        + """

Source: PROBAST+AI, <https://www.probast.org/probast-ai/>.
"""
    )


def generate_probast_ai(output_path: str | Path = "docs/probast-ai.md") -> Path:
    """Write the PROBAST+AI self-assessment."""

    return write_markdown(output_path, render_probast_ai())
