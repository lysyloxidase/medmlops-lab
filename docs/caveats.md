# Caveats

Project caveats:

1. This is a portfolio demo, not a medical device.
2. The default dataset is historical and not representative of all care
   settings.
3. The target is a proxy outcome and not a direct measure of care quality.
4. The positive class is imbalanced.
5. Diabetes 130 lacks a clean date field for temporal splitting.
6. Identifier columns must never be predictors.
7. Race, gender, and age are audit-only by default.
8. Missingness is encoded partly as `?`.
9. Diagnosis codes need careful grouping before modeling.
10. CPU determinism is scoped to a pinned environment and architecture.
11. External dependency releases can change behavior.
12. Local MinIO is a development stand-in for managed object storage.
13. Validation contracts are necessary but not sufficient for clinical safety.
14. Conformal prediction coverage is marginal, not conditional: it holds on
    average under exchangeability assumptions, not necessarily for every
    subgroup, and can break under drift.
15. Net benefit depends on the chosen clinical threshold probabilities and does
    not establish clinical utility by itself.
16. Future monitoring requires production-like data.
17. Any real-world use would require independent clinical, legal, security, and
    regulatory review.
