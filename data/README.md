# Data Directory

Data artifacts are produced and versioned through DVC. Raw, interim, and
processed datasets are intentionally ignored by git.

## `raw/`

Original downloads. Phase 1 downloads Diabetes 130-US Hospitals from the UCI
Machine Learning Repository via `ucimlrepo`.

## `interim/`

Validated but not feature-engineered artifacts. Phase 1 writes
`validated.parquet` here after Pandera and statistical contracts pass.

## `processed/`

Feature-engineered datasets introduced in Phase 2.

## Licensing

The default Diabetes 130-US Hospitals dataset is redistributable under CC BY 4.0
with citation to Strack et al. Any additional dataset must include license,
source, citation, and governance notes before ingestion.
