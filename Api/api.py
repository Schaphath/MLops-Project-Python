"""API d'inférence OncoScan.

Charge le pipeline scikit-learn au démarrage et expose :
    GET  /         infos de base
    GET  /health   état du service (utilisé par le healthcheck Docker)
    GET  /model    fiche du modèle servi (métriques, seuil, date d'entraînement)
    POST /predict  prédiction, protégée par la clé API (en-tête X-API-Key)
    GET  /metrics  métriques Prometheus
"""

import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
import sklearn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security, status
from fastapi.security import APIKeyHeader
from prometheus_client import Counter, Histogram, Info
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

# En local, on lit le .env à la racine du projet. Dans Docker il n'existe pas
# (exclu par .dockerignore) et les variables viennent du docker-compose.
# override=False : une variable déjà définie n'est jamais écrasée.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("oncoscan_api")


# Configuration
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "Save_models" / "oncoscan_pipeline.joblib"

# Un chemin relatif est pris depuis la racine du projet, pas depuis le dossier
# où l'on a lancé uvicorn.
_model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
MODEL_PATH = _model_path if _model_path.is_absolute() else PROJECT_DIR / _model_path
MODEL_VERSION = os.getenv("MODEL_VERSION") or os.getenv("VERSION") or "dev"

API_KEY = os.getenv("API_KEY")

# Même liste et même ordre qu'à l'entraînement (Model/train.py)
FEATURE_ORDER = ["perimeter", "concave_points", "texture", "smoothness", "symmetry"]


# Schémas
class PredictionRequest(BaseModel):
    """Les 5 mesures du noyau cellulaire (valeurs « worst » du jeu WDBC)."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "examples": [
                {
                    "perimeter": 87.46,
                    "concave_points": 0.04781,
                    "texture": 17.77,
                    "smoothness": 0.08474,
                    "symmetry": 0.1812,
                }
            ]
        },
    )

    # Bornes larges autour des valeurs vues à l'entraînement.
    # Elles sont reprises telles quelles dans Database/schema.sql et Interface/app.py.
    perimeter: float = Field(..., gt=40.0, le=250.0)
    concave_points: float = Field(..., ge=0.0, le=0.3)
    texture: float = Field(..., gt=5.0, le=50.0)
    smoothness: float = Field(..., gt=0.01, le=0.3)
    symmetry: float = Field(..., gt=0.05, le=0.7)


class PredictionResponse(BaseModel):
    prediction: Literal["M", "B"]
    label: Literal["Maligne", "Bénigne"]
    probability_malignant: float | None = Field(default=None, ge=0.0, le=1.0)
    model_version: str = MODEL_VERSION


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    pipeline_loaded: bool
    model_version: str = MODEL_VERSION


# Métriques Prometheus propres au modèle (aucune donnée patient, que des agrégats)
PREDICTIONS_TOTAL = Counter(
    "oncoscan_predictions_total",
    "Prédictions servies, par classe prédite",
    ["prediction", "model_version"],
)
PROBABILITY_MALIGNANT = Histogram(
    "oncoscan_probability_malignant",
    "Probabilité de malignité renvoyée",
    ["model_version"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)
INFERENCE_SECONDS = Histogram(
    "oncoscan_inference_seconds",
    "Temps de calcul du modèle seul (hors réseau et validation)",
    ["model_version"],
    buckets=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
MODEL_INFO = Info("oncoscan_model", "Modèle actuellement servi par l'API")


# Chargement du modèle
def load_pipeline(path: Path):
    if not path.is_file():
        raise RuntimeError(f"Modèle introuvable : {path}")
    try:
        return joblib.load(path)
    except Exception as error:
        raise RuntimeError(f"Impossible de charger {path.name}") from error


def load_metadata(model_path: Path) -> dict[str, Any]:
    """Lit la fiche JSON écrite par train.py à côté du modèle.

    Elle n'est pas indispensable pour prédire : si elle manque, on le signale
    et l'API démarre quand même.
    """
    path = model_path.with_suffix(".json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Fiche du modèle absente ou illisible : %s", path)
        return {}


def find_positive_index(pipeline) -> int | None:
    """Position de la classe « maligne » dans predict_proba."""
    if not hasattr(pipeline, "predict_proba"):
        return None
    classes = list(pipeline.classes_)
    for positive in (1, "M"):
        if positive in classes:
            return classes.index(positive)
    # Sans ce message, l'API renverrait des prédictions sans probabilité sans que personne ne s'en rende compte
    logger.warning("Classe maligne introuvable dans %s : aucune probabilité ne sera renvoyée.", classes)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY:
        raise RuntimeError("La variable API_KEY n'est pas définie")

    # La version de scikit-learn est loguée exprès : un écart avec celle de
    # l'entraînement est la première chose à vérifier si les prédictions dérapent.
    logger.info("Chargement du modèle : %s (scikit-learn %s)", MODEL_PATH, sklearn.__version__)
    app.state.pipeline = load_pipeline(MODEL_PATH)
    app.state.positive_idx = find_positive_index(app.state.pipeline)
    app.state.metadata = load_metadata(MODEL_PATH)

    MODEL_INFO.info({
        "version": MODEL_VERSION,
        "algorithm": str(app.state.metadata.get("model", "inconnu")),
        "calibration": str(app.state.metadata.get("calibration", "inconnue")),
        "threshold": str(app.state.metadata.get("decision_threshold", "")),
        "trained_at": str(app.state.metadata.get("trained_at", "")),
    })
    logger.info("Modèle prêt (version %s).", MODEL_VERSION)
    yield
    app.state.pipeline = None


app = FastAPI(title="OncoScan API", version=MODEL_VERSION, lifespan=lifespan)

# /metrics n'est pas protégé par la clé : le port 8000 n'est ouvert que sur le
# réseau Docker et sur 127.0.0.1. On exclut /health, appelé toutes les 30 s par
# le healthcheck, pour ne pas noyer les vraies requêtes dans les statistiques.
Instrumentator(excluded_handlers=["/metrics", "/health"]).instrument(app).expose(
    app, include_in_schema=False
)


# Authentification
# APIKeyHeader ajoute le bouton « Authorize » dans /docs
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(x_api_key: str | None = Security(api_key_header)) -> None:
    # compare_digest : comparaison en temps constant, la durée ne trahit pas la clé
    is_valid = (
        bool(API_KEY)
        and bool(x_api_key)
        and secrets.compare_digest(x_api_key.encode(), API_KEY.encode())
    )
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide ou manquante",
        )


# Endpoints
@app.get("/", tags=["Info"])
def info() -> dict[str, str]:
    return {"name": "OncoScan API", "version": MODEL_VERSION, "status": "online"}


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health(request: Request, response: Response) -> HealthResponse:
    pipeline_loaded = getattr(request.app.state, "pipeline", None) is not None
    if not pipeline_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="healthy" if pipeline_loaded else "unhealthy",
        pipeline_loaded=pipeline_loaded,
    )


@app.get("/model", tags=["Info"])
def model_card(request: Request) -> dict[str, Any]:
    """Ce que l'on sait du modèle servi : algorithme, seuil, scores sur le jeu de test."""
    return {"model_version": MODEL_VERSION, **getattr(request.app.state, "metadata", {})}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    dependencies=[Depends(require_api_key)],
)
def predict(data: PredictionRequest, request: Request) -> PredictionResponse:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non disponible",
        )

    start = time.perf_counter()
    try:
        features = pd.DataFrame([data.model_dump()])[FEATURE_ORDER]
        is_malignant = pipeline.predict(features)[0] in (1, "M")

        probability = None
        positive_idx = getattr(request.app.state, "positive_idx", None)
        if positive_idx is not None:
            probability = round(float(pipeline.predict_proba(features)[0][positive_idx]), 4)
    except Exception as error:
        logger.exception("Erreur pendant l'inférence")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors de la prédiction",
        ) from error

    elapsed = time.perf_counter() - start
    prediction = "M" if is_malignant else "B"

    PREDICTIONS_TOTAL.labels(prediction=prediction, model_version=MODEL_VERSION).inc()
    INFERENCE_SECONDS.labels(model_version=MODEL_VERSION).observe(elapsed)
    if probability is not None:
        PROBABILITY_MALIGNANT.labels(model_version=MODEL_VERSION).observe(probability)

    # Une ligne par prédiction, sans aucune donnée patient
    logger.info(
        "prediction=%s proba=%s latency_ms=%.1f version=%s",
        prediction, probability, elapsed * 1000, MODEL_VERSION,
    )

    return PredictionResponse(
        prediction=prediction,
        label="Maligne" if is_malignant else "Bénigne",
        probability_malignant=probability,
    )
