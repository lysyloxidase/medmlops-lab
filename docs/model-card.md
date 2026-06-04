# MedMLOps-Lab Model Card

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

- Hero test AUROC: `0.6815950902`
- Hero test AUPRC: `0.22916623`
- Calibrated test AUROC: `0.6807271685`
- Calibrated test AUPRC: `0.2195125318`
- Brier score: `0.094046543`
- ECE: `0.0038682659`
- Calibration slope: `0.9811621365`
- Calibration intercept: `-0.0371014336`
- Calibration method: `isotonic`
- Conformal marginal coverage: `0.9027709541`
- Conformal abstention rate: `0.0411221382`

Evidence: `reports/train_metrics.json`, `reports/calibration_metrics.json`,
`reports/conformal_metrics.json`, and `reports/clinical_metrics.json`.

## Fairness

`race, gender, age` are retained for **audit only** and are not predictors. The audit
reports demographic parity, equalized odds, equal opportunity, within-group
calibration, AUROC, and AUPRC. Race is treated as a social and political
construct, not a biological correction factor.

| Audit-only feature | AUROC gap | AUPRC gap | ECE gap |
|---|---:|---:|---:|
| race | 0.1570763672700719 | 0.25714119999652607 | 0.023574109283502634 |
| gender | 0.007397917472156235 | 0.013587107404677445 | 0.0012085337297361174 |
| age | 0.24078395357418958 | 0.31797894019397954 | 0.03343860360596816 |

When outcome base rates differ across groups, a calibrated score generally cannot also satisfy equalized odds except in special cases. This audit exposes the trade-off; it does not claim that the model is fully fair.

Race is a social and political construct, not a biological correction factor. The removal of race coefficients from eGFR equations is a cautionary lesson: race is retained here only to audit inequity and is never used as a predictor.

Evidence: `reports/fairness.json` and `docs/adr/0003-race-as-audit-only.md`.

## Monitoring And Safety

The API abstains on ambiguous or empty conformal sets. Monitoring includes
Evidently reports, transparent PSI/KS/Wasserstein/MMD calculations, and delayed
label performance checks. Synthetic drift demonstrations are not evidence of
observed real-world drift. Delayed-label degradation is retrospective.

- Latest delayed-label status: `INSUFFICIENT_LABELS`
- AUROC floor alert: `False`

## Limitations

Conformal coverage is marginal, not subgroup-conditional, and can fail under
distribution shift. Fairness metrics may conflict and do not prove absence of
harm. Missing subgroup data and small groups can make estimates unstable.
Retrospective performance and calibration do not establish clinical utility.

## Reproducibility

- Git commit: `655a062de15deb801e08627571cee00ed1c50850`
- `uv.lock` SHA-256: `616f43e316f9655558071f247087f0c3f3639a04b0d8332b49a6363c16cee88b`
- `dvc.lock` SHA-256 at generation time: `688fe9ed0c526e493a5be103eac9416f4382165c54634e6a7ee680a8d1ddedd3`
