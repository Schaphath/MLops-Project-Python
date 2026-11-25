# syntax=docker/dockerfile:1.7

#---------------#
#  BUILD STAGE  #
#---------------#
# Étape 1 : on installe les dépendances Python dans un environnement virtuel.
# Cette étape ne sera PAS dans l'image finale (image plus légère et plus sûre).
FROM python:3.12.15-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Création d'un venv pour isoler les dépendances de l'API
RUN python -m venv "$VIRTUAL_ENV" && \
    pip install --upgrade pip

# Copie des dépendances & installation de requirements_api.txt  
COPY Api/requirements_api.txt ./requirements_api.txt
RUN pip install -r requirements_api.txt

#-----------------#
#  RUNTIME STAGE  #
#-----------------#
FROM python:3.12.15-slim-bookworm AS runtime

# Version passée au build (docker compose build ou --build-arg VERSION=...)
ARG VERSION=dev

LABEL org.opencontainers.image.title="oncoscan-api" \
      org.opencontainers.image.description="API FastAPI OncoScan (modèle inclus)" \
      org.opencontainers.image.version="${VERSION}"

# MODEL_PATH et MODEL_VERSION sont dans l'image : le modèle est figé par le tag d'image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=/app/Save_models/oncoscan_pipeline.joblib \
    MODEL_VERSION=${VERSION}

# Bibliothèques système + utilisateur non-root (UID/GID fixes)
# upgrade : applique les correctifs de sécurité Debian sortis après la création
# de l'image de base (sinon Trivy bloque la CI sur des failles déjà corrigées)
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 10001 app && \
    useradd --uid 10001 --gid app --create-home --home-dir /home/app app

# Venv de l'étape 1 (propriétaire root : l'utilisateur app ne peut pas le modifier)
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app/api

# Code de l'API (le .dockerignore de la racine écarte les fichiers inutiles ou sensibles)
COPY Api/ /app/api/

# Le modèle fait partie de l'image : un tag d'image = un modèle précis.
COPY Save_models/oncoscan_pipeline.joblib Save_models/oncoscan_pipeline.json /app/Save_models/

# Plus aucun droit root à partir d'ici
USER 10001:10001

# Jusqu'à 2 min pour démarrer (machines lentes) ; un check réussi passe "healthy" tout de suite
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

EXPOSE 8000

STOPSIGNAL SIGTERM

# Lancement de l'API FastAPI avec uvicorn (serveur ASGI)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]