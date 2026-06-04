"""Generate the evidence-linked MedMLOps-Lab model card."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from medmlops.governance.common import nested, provenance_lines, write_markdown


def _fairness_gap_rows(fairness: dict[str, Any]) -> str:
    features = fairness.get("features", {})
    if not isinstance(features, dict):
        return "| not available | not available | not available | not available |"
    rows: list[str] = []
    for feature, metrics in features.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            f"| {feature} | {metrics.get('subgroup_auroc_gap', 'not available')} "
            f"| {metrics.get('subgroup_auprc_gap', 'not available')} "
            f"| {metrics.get('calibration_ece_gap', 'not available')} |"
        )
    return "\n".join(rows)


def render_model_card(evidence: dict[str, dict[str, Any]]) -> str:
    """Render a model card from pipeline reports."""

    clinical = evidence["clinical_metrics"]
    calibration = evidence["calibration_metrics"]
    conformal = evidence["conformal_metrics"]
    fairness = evidence["fairness"]
    train = evidence["train_metrics"]
    monitoring = evidence["performance_monitoring"]
    provenance = "\n".join(provenance_lines())
    sensitive = nested(
        fairness, "audit_scope", "sensitive_features_audit_only", default=[]
    )
    if isinstance(sensitive, list):
        sensitive = ", ".join(str(value) for value in sensitive)
    fairness_gap_rows = _fairness_gap_rows(fairness)
    return f"""# MedMLOps-Lab Model Card

> **Research and portfolio demonstration only. Not a medical device, not for
> clinical decision-making, and not a regulatory certification.** The governance
> mappings below are aspirational self-assessments.

## Model Details

- Task: predict 30-day hospital readmission risk for the Diabetes 130-US
  Hospitals dataset.
- Hero model: CPU-trained PyTorch tabular MLP with isotonic calibration and a
  MAPIE conformal abstention gate.
- Predictor boundary: audit-only attributes are excluded from model inputs.

## Intended Use

Reproducible education, MLOps experimentation, and evaluation of clinical-risk
model safeguards. Any real clinical use would require prospective validation,
human-factors work, safety engineering, local governance, and regulatory review.

## Out Of Scope

Diagnosis, treatment selection, autonomous triage, bedside deployment, or claims
of effectiveness for current patients. The historical US dataset does not
establish transportability to another institution, era, or population.

## Training And Evaluation

- Hero test AUROC: `{nested(train, "hero", "test", "auroc")}`
- Hero test AUPRC: `{nested(train, "hero", "test", "auprc")}`
- Calibrated test AUROC: `{nested(clinical, "auroc")}`
- Calibrated test AUPRC: `{nested(clinical, "auprc")}`
- Brier score: `{nested(clinical, "brier")}`
- ECE: `{nested(clinical, "ece")}`
- Calibration slope: `{nested(clinical, "calibration_slope")}`
- Calibration intercept: `{nested(clinical, "calibration_intercept")}`
- Calibration method: `{nested(calibration, "method")}`
- Conformal marginal coverage: `{nested(conformal, "marginal_coverage")}`
- Conformal abstention rate: `{nested(conformal, "abstention_rate")}`

Evidence: `reports/train_metrics.json`, `reports/calibration_metrics.json`,
`reports/conformal_metrics.json`, and `reports/clinical_metrics.json`.

## Fairness

`{sensitive}` are retained for **audit only** and are not predictors. The audit
reports demographic parity, equalized odds, equal opportunity, within-group
calibration, AUROC, and AUPRC. Race is treated as a social and political
construct, not a biological correction factor.

| Audit-only feature | AUROC gap | AUPRC gap | ECE gap |
|---|---:|---:|---:|
{fairness_gap_rows}

{nested(fairness, "impossibility_tradeoff")}

{nested(fairness, "egfr_race_lesson")}

Evidence: `reports/fairness.json` and `docs/adr/0003-race-as-audit-only.md`.

## Monitoring And Safety

The API abstains on ambiguous or empty conformal sets. Monitoring includes
Evidently reports, transparent PSI/KS/Wasserstein/MMD calculations, and delayed
label performance checks. Synthetic drift demonstrations are not evidence of
observed real-world drift. Delayed-label degradation is retrospective.

- Latest delayed-label status: `{nested(monitoring, "status")}`
- AUROC floor alert: `{nested(monitoring, "alert")}`

## Limitations

Conformal coverage is marginal, not subgroup-conditional, and can fail under
distribution shift. Fairness metrics may conflict and do not prove absence of
harm. Missing subgroup data and small groups can make estimates unstable.
Retrospective performance and calibration do not establish clinical utility.

## Reproducibility

{provenance}
"""


def generate_model_card(
    evidence: dict[str, dict[str, Any]],
    output_path: str | Path = "docs/model-card.md",
) -> Path:
    """Write the model card."""

    return write_markdown(output_path, render_model_card(evidence))
