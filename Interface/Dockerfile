# syntax=docker/dockerfile:1.7

# Version de Python écrite en clair dans les deux FROM (pas d'ARG) :
# c'est la seule façon pour que Dependabot puisse la mettre à jour.

#---------------#
#  BUILD STAGE  #
#---------------#
# Étape 1 : dépendances dans un environnement virtuel.
# L'image slim n'a PAS de compilateur : utiliser psycopg2-binary, pas psycopg2.
FROM python:3.12.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /build

RUN python -m venv "$VIRTUAL_ENV" && \
    pip install --upgrade pip

COPY Interface/requirements_interface.txt ./requirements_interface.txt
RUN pip install -r requirements_interface.txt

#-----------------#
#  RUNTIME STAGE  #
#-----------------#
FROM python:3.12.12-slim-bookworm AS runtime

ARG VERSION=2.0.0

LABEL org.opencontainers.image.title="oncoscan-app" \
      org.opencontainers.image.description="Interface web Streamlit OncoScan" \
      org.opencontainers.image.version="${VERSION}"

# Prod : TOOLBAR_MODE=minimal masque le menu développeur (Deploy, Rerun...),
# SHOW_ERROR_DETAILS=false n'affiche jamais de traceback aux utilisateurs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_VERSION=${VERSION} \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_CLIENT_TOOLBAR_MODE=minimal \
    STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=false

# Bibliothèques système + utilisateur non-root (UID/GID fixes)
# upgrade : applique les correctifs de sécurité Debian sortis après la création
# de l'image de base (sinon Trivy bloque la CI sur des failles déjà corrigées)
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libstdc++6 \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 10001 app && \
    useradd --uid 10001 --gid app --create-home --home-dir /home/app app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app/interface

# Code de l'interface (app.py, style.css, favicon.svg)
COPY Interface/ /app/interface/

USER 10001:10001

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)"]

EXPOSE 8501

STOPSIGNAL SIGTERM

CMD ["streamlit", "run", "app.py"]