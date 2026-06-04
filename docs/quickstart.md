# Quickstart

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Docker.

```bash
git clone https://github.com/lysyloxidase/medmlops-lab
cd medmlops-lab
make setup
make test
make reproduce
make serve
```

`make reproduce` downloads the public Diabetes 130-US Hospitals dataset,
rebuilds all DVC stages, and verifies canonical metric hashes. It intentionally
fails when the produced metrics diverge from the committed reference.

`make serve` launches:

- FastAPI at `http://localhost:8000`
- MLflow at `http://localhost:5000`
- MinIO at `http://localhost:9001`
- PostgreSQL for append-only prediction audit records

Useful commands:

```bash
make data             # ingest, validate, preprocess, split
make train            # model training stage
make clinical         # calibration, conformalization, evaluation
make monitor          # drift and delayed-label monitoring
make responsible-ai   # fairness audit and governance documents
make docs             # local MkDocs site
```

The project is a portfolio demonstration, not a medical device.
