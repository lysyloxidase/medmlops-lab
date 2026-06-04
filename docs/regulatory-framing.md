# Regulatory Framing

> **Aspirational compliance mapping only. This project is not a medical device,
> has not been submitted to a regulator, and is not certified or approved.**
> Regulatory classification depends on intended purpose, claims, deployment,
> jurisdiction, and facts beyond this repository.

## FDA / IMDRF Good Machine Learning Practice

The repository maps its engineering controls to the ten GMLP guiding principles:

1. Multi-disciplinary expertise is leveraged throughout the total product life cycle.
2. Good software engineering and security practices are implemented.
3. Clinical study participants and data sets are representative of the intended patient population.
4. Training data sets are independent of test sets.
5. Selected reference data sets are based upon best available methods.
6. Model design is tailored to the available data and reflects the intended use.
7. Focus is placed on the performance of the human-AI team.
8. Testing demonstrates device performance during clinically relevant conditions.
9. Users are provided clear, essential information.
10. Deployed models are monitored for performance and retraining risks are managed.

Relevant evidence includes independent data splits, DVC provenance, calibration,
conformal abstention, subgroup auditing, monitoring, and generated governance
documents. Important gaps include clinical workflow evaluation, representative
prospective studies, human-AI team testing, and a regulated quality system.

Source: FDA, "Good Machine Learning Practice for Medical Device Development:
Guiding Principles", <https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles>.

## EU AI Act

This research demonstration is not deployed and makes no medical-device claim.
An AI system used as a safety component of a regulated medical device, or itself
covered by relevant product-safety legislation and requiring third-party
conformity assessment, may be classified as **high-risk** under the EU AI Act.
Such use would bring obligations around risk management, data governance,
technical documentation, logging, transparency, human oversight, accuracy,
robustness, cybersecurity, post-market monitoring, and incident handling.

This is a cautious project framing, not legal advice or a classification
determination. Source: European Commission, "AI Act",
<https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai>.
