# Data Dictionary

Default dataset: Diabetes 130-US Hospitals for Years 1999-2008.

Key fields:

- `encounter_id`: encounter identifier, excluded from predictors
- `patient_nbr`: patient identifier, excluded from predictors
- `race`: audit-only demographic field
- `gender`: audit-only demographic field
- `age`: audit-only age bracket
- `time_in_hospital`: length of stay in days, expected range 1-14
- `num_lab_procedures`: count of lab procedures
- `num_medications`: count of medications
- `number_diagnoses`: number of diagnoses
- `readmitted`: original target with values `NO`, `>30`, `<30`
- `readmitted_30d`: derived binary target, positive when `readmitted == "<30"`

Full raw-column presence is enforced in `medmlops.data.contracts`.
