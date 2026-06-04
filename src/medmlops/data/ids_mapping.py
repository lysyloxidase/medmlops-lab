"""Human-readable mappings for Diabetes 130 ID-coded fields."""

from __future__ import annotations

ADMISSION_TYPE_ID = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Not available",
    6: "NULL",
    7: "Trauma center",
    8: "Not mapped",
}

DISCHARGE_DISPOSITION_ID = {
    1: "Discharged to home",
    2: "Transferred to another short-term hospital",
    3: "Transferred to skilled nursing facility",
    4: "Transferred to intermediate care facility",
    5: "Transferred to another inpatient care institution",
    6: "Discharged to home with home health service",
    7: "Left against medical advice",
    11: "Expired",
    18: "NULL",
    30: "Still patient or expected to return for outpatient services",
}

ADMISSION_SOURCE_ID = {
    1: "Physician referral",
    2: "Clinic referral",
    3: "HMO referral",
    4: "Transfer from hospital",
    5: "Transfer from skilled nursing facility",
    6: "Transfer from another health care facility",
    7: "Emergency room",
    8: "Court/law enforcement",
    9: "Not available",
    17: "NULL",
    20: "Not mapped",
    25: "Transfer from ambulatory surgery center",
}


def group_icd9_diagnosis(code: object) -> str:
    """Map an ICD-9 diagnosis code into a coarse clinical category."""

    if code is None:
        return "missing"

    value = str(code).strip()
    if value in {"", "?", "nan", "None", "null"}:
        return "missing"
    if value.startswith(("V", "E")):
        return "supplemental"

    try:
        numeric = float(value)
    except ValueError:
        return "other"

    if 390 <= numeric <= 459 or numeric == 785:
        return "circulatory"
    if 460 <= numeric <= 519 or numeric == 786:
        return "respiratory"
    if 520 <= numeric <= 579 or numeric == 787:
        return "digestive"
    if int(numeric) == 250:
        return "diabetes"
    if 800 <= numeric <= 999:
        return "injury"
    if 710 <= numeric <= 739:
        return "musculoskeletal"
    if 580 <= numeric <= 629 or numeric == 788:
        return "genitourinary"
    if 140 <= numeric <= 239:
        return "neoplasms"
    return "other"
