# Tests de validation du modèle : données, cohérence de l'artefact, performances, robustesse.
# python -m pytest Tests/test_model.py -v --json-report --json-report-file=Tests_results/results.json


import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "Data" / "process" / "cancer_select.csv"
# Nom fixe produit par Model/train.py
DEFAULT_MODEL_PATH = BASE_DIR / "Save_models" / "oncoscan_pipeline.joblib"

DATA_PATH = Path(os.getenv("TEST_DATA_PATH", str(DEFAULT_DATA_PATH)))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

SELECTED_FEATURES = [
    "perimeter",
    "concave_points",
    "texture",
    "smoothness",
    "symmetry",
]

# Seuils minimaux attendus sur le jeu de test (les mêmes que dans train.py)
SEUIL_RECALL = 0.95  # on veut rater le moins de tumeurs malignes possible
SEUIL_PRECISION = 0.90
SEUIL_AUC = 0.95
SEUIL_F1 = 0.90
# Fiabilité des probabilités affichées aux médecins (plus bas = mieux).
# L'AdaBoost non calibré était autour de 0.08 : on refuse de revenir à ce niveau.
SEUIL_BRIER = 0.07

# Doit rester identique au split fait à l'entraînement, sinon on évalue sur des données vues
TEST_SIZE = 0.2
RANDOM_STATE = 42


def _resolve_repo_path(path: Path) -> Path:
    """Chemin relatif -> relatif à la racine du projet."""
    return path if path.is_absolute() else BASE_DIR / path


def _normalize_target(series: pd.Series) -> pd.Series:
    """Convertit la cible en 0/1 (M=1, B=0)."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    mapped = series.map({"M": 1, "B": 0})
    assert mapped.notna().all(), f"Valeurs de cible inattendues : {set(series[mapped.isna()])}"
    return mapped.astype(int)


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    """Charge le dataset et vérifie les colonnes attendues."""
    data_path = _resolve_repo_path(DATA_PATH)
    assert data_path.exists(), f"Dataset introuvable : {data_path}"

    df = pd.read_csv(data_path)
    assert "diagnosis" in df.columns, "Colonne 'diagnosis' absente."
    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    assert not missing, f"Features manquantes : {missing}"
    return df


@pytest.fixture(scope="module")
def model_path() -> Path:
    path = _resolve_repo_path(MODEL_PATH)
    assert path.exists(), f"Modèle introuvable : {path} (lancer Model/train.py)"
    return path


@pytest.fixture(scope="module")
def pipeline(model_path):
    """Charge le modèle final (calibré + seuil de décision)."""
    return joblib.load(model_path)


@pytest.fixture(scope="module")
def metadata(model_path) -> dict:
    """Fiche JSON écrite par train.py à côté du modèle."""
    path = model_path.with_suffix(".json")
    assert path.exists(), f"Fiche de métadonnées introuvable : {path}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def predictions(dataset, pipeline):
    """Refait le split de test et calcule les prédictions."""
    X = dataset[SELECTED_FEATURES]
    y = _normalize_target(dataset["diagnosis"])

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    return {
        "y_test": y_test.to_numpy(),
        "y_pred": np.asarray(pipeline.predict(X_test)).astype(int),
        # Le modèle final doit toujours fournir des probabilités (affichées dans l'appli)
        "y_proba": pipeline.predict_proba(X_test)[:, 1],
        "X_test": X_test,
    }


class TestQualiteDonnees:
    def test_integrite_donnees(self, dataset):
        assert len(dataset) >= 100, "Dataset trop petit."
        cols = SELECTED_FEATURES + ["diagnosis"]
        assert dataset[cols].isna().sum().sum() == 0, "Valeurs manquantes détectées."

    def test_encodage_cible(self, dataset):
        classes = set(dataset["diagnosis"].dropna().unique())
        assert classes in ({"M", "B"}, {0, 1}), f"Classes inattendues : {classes}"


class TestCoherenceArtefact:
    """Vérifie que le modèle et sa fiche correspondent à ce que l'API attend."""

    def test_version_sklearn(self, metadata):
        # Évite le warning InconsistentVersionWarning (et des prédictions potentiellement fausses)
        assert metadata["sklearn_version"] == sklearn.__version__, (
            f"Modèle entraîné avec scikit-learn {metadata['sklearn_version']}, "
            f"testé avec {sklearn.__version__}"
        )

    def test_features_identiques(self, metadata):
        # Même liste et même ordre que l'API (FEATURE_ORDER)
        assert metadata["features"] == SELECTED_FEATURES, f"Features du modèle : {metadata['features']}"

    def test_seuil_decision(self, pipeline, metadata):
        threshold = getattr(pipeline, "threshold", None)
        assert threshold is not None, "Le modèle n'a pas de seuil de décision (FixedThresholdClassifier attendu)."
        assert 0.0 < threshold < 1.0, f"Seuil hors de ]0, 1[ : {threshold}"
        assert threshold == pytest.approx(metadata["decision_threshold"], abs=1e-4), (
            "Le seuil du modèle ne correspond pas à celui de la fiche."
        )

    def test_classes_attendues(self, pipeline):
        # L'API cherche la classe 1 dans classes_ pour renvoyer la probabilité de malignité
        assert list(pipeline.classes_) == [0, 1], f"classes_ inattendues : {list(pipeline.classes_)}"


class TestMetriquesModele:
    def test_recall_critique_medical(self, predictions):
        recall = recall_score(predictions["y_test"], predictions["y_pred"])
        assert recall >= SEUIL_RECALL, f"Recall insuffisant : {recall:.3f}"

    def test_precision(self, predictions):
        precision = precision_score(predictions["y_test"], predictions["y_pred"])
        assert precision >= SEUIL_PRECISION, f"Précision insuffisante : {precision:.3f}"

    def test_f1_score(self, predictions):
        f1 = f1_score(predictions["y_test"], predictions["y_pred"])
        assert f1 >= SEUIL_F1, f"F1-score insuffisant : {f1:.3f}"

    def test_auc_score(self, predictions):
        auc = roc_auc_score(predictions["y_test"], predictions["y_proba"])
        assert auc >= SEUIL_AUC, f"AUC insuffisant : {auc:.3f}"

    def test_calibration_brier(self, predictions):
        brier = brier_score_loss(predictions["y_test"], predictions["y_proba"])
        assert brier <= SEUIL_BRIER, f"Probabilités mal calibrées : Brier = {brier:.4f}"


class TestRobustesseInference:
    def test_probabilites_valides(self, predictions):
        proba = predictions["y_proba"]
        assert np.all((proba >= 0.0) & (proba <= 1.0)), "Probabilités hors de [0, 1]."

    def test_prediction_coherente_avec_seuil(self, pipeline, predictions):
        # L'appli affiche la classe ET la probabilité : elles doivent dire la même chose
        attendu = (predictions["y_proba"] >= pipeline.threshold).astype(int)
        np.testing.assert_array_equal(
            predictions["y_pred"], attendu,
            err_msg="La classe prédite ne correspond pas à proba >= seuil.",
        )

    def test_inference_mono_exemple(self, pipeline, predictions):
        single_sample = predictions["X_test"].iloc[[0]]
        pred = pipeline.predict(single_sample)
        assert len(pred) == 1 and int(pred[0]) in (0, 1), "Format de prédiction invalide."

    def test_determinisme_prediction(self, pipeline, predictions):
        pred_a = pipeline.predict(predictions["X_test"])
        pred_b = pipeline.predict(predictions["X_test"])
        np.testing.assert_array_equal(pred_a, pred_b, err_msg="Prédictions non déterministes.")

    def test_colonnes_manquantes_refusees(self, pipeline, predictions):
        X_incomplet = predictions["X_test"].drop(columns=[SELECTED_FEATURES[0]])
        # scikit-learn vérifie les noms de colonnes vus au fit
        with pytest.raises(ValueError):
            pipeline.predict(X_incomplet)