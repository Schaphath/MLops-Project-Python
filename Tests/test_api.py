"""Tests de l'API FastAPI (Api/api.py), sans Docker ni serveur.

Le TestClient de FastAPI appelle l'application directement et déclenche le
démarrage (chargement du vrai modèle de Save_models/).

python -m pytest Tests/test_api.py -v
"""

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

API_DIR = Path(__file__).resolve().parents[1] / "Api"
TEST_KEY = "cle-de-test"

VALID_SAMPLE = {
    "perimeter": 87.46,
    "concave_points": 0.04781,
    "texture": 17.77,
    "smoothness": 0.08474,
    "symmetry": 0.1812
    }

# Tumeur clairement maligne 
MALIGNANT_SAMPLE = {
    "perimeter": 184.6,
    "concave_points": 0.2654,
    "texture": 17.33,
    "smoothness": 0.1622,
    "symmetry": 0.4601
    }

@pytest.fixture(scope="module")
def api():
    """Importe api.py une seule fois, avec une clé de test.
    La clé est lue à l'import ; elle est posée avant pour que le .env local
    (lu avec override=False) ne puisse pas la remplacer.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("API_KEY", TEST_KEY)
        monkeypatch.syspath_prepend(str(API_DIR))
        yield importlib.import_module("api")


@pytest.fixture(scope="module")
def client(api):
    with TestClient(api.app) as test_client:
        yield test_client

def auth(key=TEST_KEY):
    return {"X-API-Key": key}

class TestDemarrage:
    def test_refuse_de_demarrer_sans_cle(self, api, monkeypatch):
        monkeypatch.setattr(api, "API_KEY", None)
        with pytest.raises(RuntimeError, match="API_KEY"), TestClient(api.app):
            pass

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["pipeline_loaded"] is True

    def test_fiche_du_modele(self, client):
        card = client.get("/model").json()
        assert card["features"] == list(VALID_SAMPLE)
        assert 0 < card["decision_threshold"] < 1

class TestAuthentification:
    def test_sans_cle(self, client):
        assert client.post("/predict", json=VALID_SAMPLE).status_code == 401

    def test_mauvaise_cle(self, client):
        response = client.post("/predict", json=VALID_SAMPLE, headers=auth("mauvaise-cle"))
        assert response.status_code == 401

class TestValidationEntrees:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("perimeter", 40.0),      # borne basse exclue
            ("perimeter", 300.0),
            ("concave_points", -0.01),
            ("texture", 51.0),
            ("smoothness", 0.0),
            ("symmetry", 0.8)
        ],
    )
    def test_valeur_hors_bornes(self, client, field, value):
        payload = {**VALID_SAMPLE, field: value}
        assert client.post("/predict", json=payload, headers=auth()).status_code == 422

    def test_champ_manquant(self, client):
        payload = {k: v for k, v in VALID_SAMPLE.items() if k != "texture"}
        assert client.post("/predict", json=payload, headers=auth()).status_code == 422

    def test_champ_inconnu(self, client):
        payload = {**VALID_SAMPLE, "age": 54}
        assert client.post("/predict", json=payload, headers=auth()).status_code == 422

    def test_texte_au_lieu_d_un_nombre(self, client):
        payload = {**VALID_SAMPLE, "perimeter": "beaucoup"}
        assert client.post("/predict", json=payload, headers=auth()).status_code == 422

class TestPrediction:
    def test_format_de_reponse(self, client):
        response = client.post("/predict", json=VALID_SAMPLE, headers=auth())
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] in ("M", "B")
        assert body["label"] == ("Maligne" if body["prediction"] == "M" else "Bénigne")
        assert 0.0 <= body["probability_malignant"] <= 1.0

    def test_cas_benin(self, client):
        body = client.post("/predict", json=VALID_SAMPLE, headers=auth()).json()
        assert body["prediction"] == "B"

    def test_cas_malin(self, client):
        body = client.post("/predict", json=MALIGNANT_SAMPLE, headers=auth()).json()
        assert body["prediction"] == "M"
        assert body["probability_malignant"] > 0.5

class TestMetriques:
    def test_metriques_exposees(self, client):
        client.post("/predict", json=VALID_SAMPLE, headers=auth())
        text = client.get("/metrics").text
        assert "oncoscan_predictions_total" in text
        assert "oncoscan_probability_malignant_bucket" in text
        assert "oncoscan_model_info" in text
        assert 'handler="/predict"' in text

    def test_health_exclu_des_statistiques(self, client):
        client.get("/health")
        assert 'handler="/health"' not in client.get("/metrics").text
