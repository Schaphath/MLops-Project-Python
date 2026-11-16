"""API REST"""

import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import sklearn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

# Charge automatiquement le fichier .env
load_dotenv()

# Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

logger = logging.getLogger("oncoscan_api")

# Configuration
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "Save_models" / "oncoscan_pipeline.joblib"

# Un chemin relatif dans le .env dépendait du dossier d'où on lançait uvicorn.
# On le rend relatif à la racine du projet.
_model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
MODEL_PATH = _model_path if _model_path.is_absolute() else PROJECT_DIR / _model_path
MODEL_VERSION = os.getenv("MODEL_VERSION") or os.getenv("VERSION") or "2.0.0"

# Récupération de la clé API depuis le fichier .env
API_KEY = os.getenv("API_KEY")

# Variables d'entraînement (même ordre qu'au fit)
FEATURE_ORDER = [
    "perimeter",
    "concave_points",
    "texture",
    "smoothness",
    "symmetry",
]


# Schemas Pydantic
class PredictionRequest(BaseModel):
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

    # Bornes larges autour des valeurs vues à l'entraînement
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


# Helpers & Lifespan
def load_pipeline(path: Path):
    if not path.is_file():
        raise RuntimeError(f"Artefact introuvable : {path}")
    try:
        return joblib.load(path)
    except Exception as error:
        raise RuntimeError(f"Erreur lors du chargement de {path.name}") from error


def find_positive_index(pipeline) -> int | None:
    """Trouve l'index de la classe maligne dans predict_proba."""
    if not hasattr(pipeline, "predict_proba"):
        return None
    classes = list(pipeline.classes_)
    for positive in (1, "M"):
        if positive in classes:
            return classes.index(positive)
    # Sans ça, on servait des prédictions sans probabilité sans que personne le voie
    logger.warning("Classe maligne introuvable dans %s : pas de probabilité renvoyée.", classes)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Vérifie la config et charge le modèle au démarrage."""
    if not API_KEY:
        raise RuntimeError("La variable API_KEY est introuvable")

    # Utile pour repérer un écart de version sklearn entre entraînement et prod
    logger.info("Chargement du pipeline : %s (scikit-learn %s)", MODEL_PATH, sklearn.__version__)
    app.state.pipeline = load_pipeline(MODEL_PATH)
    app.state.positive_idx = find_positive_index(app.state.pipeline)
    logger.info("Pipeline chargé avec succès.")
    yield

    app.state.pipeline = None


# FastAPI App
app = FastAPI(
    title="OncoScan API",
    version=MODEL_VERSION,
    lifespan=lifespan)


# Sécurité
# APIKeyHeader ajoute le bouton "Authorize" dans Swagger (/docs)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(x_api_key: str | None = Security(api_key_header)) -> None:
    is_valid = (
        bool(API_KEY)
        and bool(x_api_key)
        and secrets.compare_digest(x_api_key.encode(), API_KEY.encode()))
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide ou manquante")


# Endpoints
@app.get("/", tags=["Info"])
def info() -> dict[str, str]:
    return {
        "name": "OncoScan API",
        "version": MODEL_VERSION,
        "status": "online",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health(request: Request, response: Response) -> HealthResponse:
    pipeline_loaded = getattr(request.app.state, "pipeline", None) is not None

    if not pipeline_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="healthy" if pipeline_loaded else "unhealthy",
        pipeline_loaded=pipeline_loaded)


# Inférence sécurisée par la clé API
# (le commentaire était entre le décorateur et la fonction, je l'ai remonté)
@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    dependencies=[Depends(require_api_key)])
def predict(data: PredictionRequest, request: Request) -> PredictionResponse:
    pipeline = getattr(request.app.state, "pipeline", None)

    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non disponible")

    start = time.perf_counter()
    try:
        features_df = pd.DataFrame([data.model_dump()])[FEATURE_ORDER]

        prediction_raw = pipeline.predict(features_df)[0]
        is_malignant = prediction_raw in (1, "M")
        pred_label = "M" if is_malignant else "B"
        human_label = "Maligne" if is_malignant else "Bénigne"

        probability = None
        positive_idx = getattr(request.app.state, "positive_idx", None)
        if positive_idx is not None:
            probas = pipeline.predict_proba(features_df)[0]
            probability = round(float(probas[positive_idx]), 4)

    except Exception as error:
        logger.exception("Erreur lors de l'inférence")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors de la prédiction",
        ) from error

    # Trace minimale pour le monitoring (pas d'info patient, juste le résultat)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "prediction=%s proba=%s latency_ms=%.1f version=%s",
        pred_label, probability, latency_ms, MODEL_VERSION)

    return PredictionResponse(
        prediction=pred_label,
        label=human_label,
        probability_malignant=probability)