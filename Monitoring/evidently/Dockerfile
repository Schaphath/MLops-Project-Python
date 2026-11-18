# syntax=docker/dockerfile:1.7
# Job de monitoring du modèle : drift (Evidently) + performance réelle (feedback médecins).
# Même structure que Api/Dockerfile : build dans un venv, image finale sans outils de build.

FROM python:3.12.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /build

RUN python -m venv "$VIRTUAL_ENV" && pip install --upgrade pip

COPY requirements_evidently.txt .

RUN pip install -r requirements_evidently.txt


FROM python:3.12.12-slim-bookworm AS runtime

ARG VERSION=dev

LABEL org.opencontainers.image.title="oncoscan-evidently" \
      org.opencontainers.image.description="Monitoring du modèle OncoScan (drift + performance)" \
      org.opencontainers.image.version="${VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    REFERENCE_PATH=/app/reference/cancer_select.csv \
    REPORTS_DIR=/app/reports

RUN apt-get update && \
    apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 10001 app && \
    useradd --uid 10001 --gid app --create-home --home-dir /home/app app && \
    mkdir -p /app/reports /app/reference && \
    chown app:app /app/reports

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

COPY drift_job.py /app/drift_job.py

USER 10001:10001

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8085/metrics', timeout=2)"]

EXPOSE 8085

STOPSIGNAL SIGTERM

CMD ["python", "drift_job.py"]
