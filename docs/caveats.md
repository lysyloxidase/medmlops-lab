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
16. The Phase 5 covariate, prior-probability, and concept drift regimes are
    synthetic injections for demonstration, not observed real-world drift.
17. Performance drift is detectable only retrospectively on the subset with
    delayed ground-truth labels; missing or selectively available labels can
    bias the monitoring result.
18. Future monitoring requires production-like data.
19. Any real-world use would require independent clinical, legal, security, and
    regulatory review.
20. Fairness definitions can conflict. With unequal outcome base rates,
    calibration and equalized odds generally cannot both hold; subgroup metrics
    are diagnostics, not proof that a model is fair.
21. Race is a social and political construct, not a biological correction
    factor. Its presence in this project is limited to auditing inequity.
22. FDA GMLP, EU AI Act, TRIPOD+AI, and PROBAST+AI mappings are aspirational
    self-assessments, not certification, approval, or legal advice.
23. Canonical metric-hash equality is an engineering reproducibility check; it
    does not establish clinical validity, safety, usefulness, or transportability.
24. Exact reproduction is scoped to the pinned Docker image and reference CPU
    architecture. PyTorch and numerical libraries can differ across platforms.
25. The fairness release gate detects widening relative to a committed baseline;
    it does not declare existing subgroup gaps acceptable or resolved.
