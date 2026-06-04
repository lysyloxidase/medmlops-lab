# TRIPOD+AI Self-Assessment

> This is an aspirational, project-level self-assessment against the 27-item
> TRIPOD+AI reporting checklist. It is not independent review or certification.

TRIPOD+AI supersedes the original TRIPOD statement for clinical prediction
models using regression or machine-learning methods. Evidence pointers identify
where this repository addresses each item; "Addressed" does not imply adequacy.

| # | Reporting item | Status | Evidence pointer |
|---:|---|---|---|
| 1 | Title identifies prediction-model study | Addressed | `README.md` |
| 2 | Structured summary | Addressed | `README.md; docs/model-card.md` |
| 3 | Background and rationale | Addressed | `README.md; docs/model-card.md` |
| 4 | Objectives | Addressed | `README.md` |
| 5 | Source of data | Addressed | `docs/datasheet.md` |
| 6 | Study setting | Addressed | `docs/datasheet.md` |
| 7 | Eligibility criteria | Addressed | `docs/datasheet.md` |
| 8 | Outcome definition | Addressed | `params.yaml; docs/datasheet.md` |
| 9 | Candidate predictors | Addressed | `params.yaml` |
| 10 | Sample size | Addressed | `reports/data_quality.json` |
| 11 | Missing data handling | Addressed | `src/medmlops/features/pipeline.py` |
| 12 | Data preparation | Addressed | `src/medmlops/features/pipeline.py` |
| 13 | Model type and specification | Addressed | `src/medmlops/models/` |
| 14 | Model-building procedures | Addressed | `src/medmlops/models/train.py` |
| 15 | Internal validation | Addressed | `dvc.yaml; reports/train_metrics.json` |
| 16 | Performance measures | Addressed | `reports/clinical_metrics.json` |
| 17 | Fairness assessment | Addressed | `reports/fairness.json` |
| 18 | Model output and thresholds | Addressed | `params.yaml; reports/clinical_metrics.json` |
| 19 | Participant flow | Addressed | `dvc.yaml; reports/data_quality.json` |
| 20 | Participant characteristics | Addressed | `docs/datasheet.md` |
| 21 | Model performance | Addressed | `reports/clinical_metrics.json` |
| 22 | Model specification availability | Addressed | `models/; src/medmlops/models/` |
| 23 | Interpretation | Addressed | `docs/model-card.md` |
| 24 | Limitations | Addressed | `docs/caveats.md; docs/model-card.md` |
| 25 | Implications and future research | Addressed | `docs/model-card.md` |
| 26 | Supplementary information and protocol | Addressed | `docs/; dvc.yaml` |
| 27 | Funding, conflicts, registration, and data/code access | Addressed | `README.md; LICENSE` |

Source: Collins GS et al., *BMJ* 2024;385:e078378,
<https://www.bmj.com/content/385/bmj-2023-078378>.
