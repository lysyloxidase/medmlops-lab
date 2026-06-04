FROM ghcr.io/astral-sh/uv:0.11.18-python3.12-trixie-slim@sha256:9f9a7c30d604c8297ac14321b91b42f8516ff207f42b76b1bf29974fad523f6e

LABEL org.opencontainers.image.title="MedMLOps-Lab"
LABEL org.opencontainers.image.description="CPU-only clinical risk MLOps reference platform"
LABEL org.opencontainers.image.source="https://github.com/lysyloxidase/medmlops-lab"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git libgomp1 make \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --all-groups --no-install-project

COPY . .
RUN printf '[core]\n    no_scm = true\n' > .dvc/config.local \
    && chmod +x docker/entrypoint.sh \
    && uv sync --frozen --all-groups

ENTRYPOINT ["sh", "/app/docker/entrypoint.sh"]
CMD ["medmlops", "--help"]
