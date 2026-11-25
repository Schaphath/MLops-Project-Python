"""Job de monitoring de notre modèle.

À intervalle régulier :
1. lit les prédictions récentes dans PostgreSQL (table predictions) ;
2. compare leurs variables aux données d'entraînement avec Evidently (data drift)
   et génère un rapport HTML ;
3. calcule la performance réelle sur les cas confirmés par les médecins (doctor_feedback) ;
4. expose le tout sur /metrics pour Prometheus, et les rapports HTML sur /.

Aucune donnée patient n'est exportée : seulement des agrégats.
"""

import html
import logging
import os
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import psycopg
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

logger = logging.getLogger("oncoscan_evidently")

# Configuration
FEATURES = ["perimeter", "concave_points", "texture", "smoothness", "symmetry"]
REFERENCE_PATH = Path(os.getenv("REFERENCE_PATH", "/app/reference/cancer_select.csv"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "/app/reports"))
WINDOW_DAYS = int(os.getenv("WINDOW_DAYS", "10")) # prédictions analysées
INTERVAL_SECONDS = int(os.getenv("INTERVAL_SECONDS", "3600"))
MIN_ROWS = int(os.getenv("MIN_ROWS", "10")) # en dessous, le drift n'a pas de sens
KEEP_REPORTS = int(os.getenv("KEEP_REPORTS", "48"))
PORT = int(os.getenv("PORT", "8085"))

DB_DSN = (
    f"host={os.getenv('DB_HOST', 'postgres')} port={os.getenv('DB_PORT', '5432')} "
    f"dbname={os.environ['DB_NAME']} user={os.environ['DB_USER']} "
    f"password={os.environ['DB_PASSWORD']} connect_timeout=10"
)

# Métriques Prometheus
LAST_RUN_TS = Gauge("oncoscan_monitoring_last_run_timestamp_seconds", "Fin du dernier calcul (epoch)")
LAST_RUN_OK = Gauge("oncoscan_monitoring_last_run_success", "1 si le dernier calcul a réussi")
CURRENT_ROWS = Gauge("oncoscan_current_rows", "Prédictions dans la fenêtre analysée")
CONFIRMED_ROWS = Gauge("oncoscan_confirmed_rows", "Prédictions confirmées par un médecin dans la fenêtre")
DRIFT_SHARE = Gauge("oncoscan_drift_share", "Part des variables en drift (0 à 1)")
DRIFT_COUNT = Gauge("oncoscan_drift_count", "Nombre de variables en drift")
FEATURE_DRIFT = Gauge("oncoscan_feature_drift_score", "Score de drift par variable (p-value ou distance)", ["feature"])
FEATURE_DRIFTED = Gauge("oncoscan_feature_drifted", "1 si la variable a dérivé", ["feature"])
FEATURE_MEAN = Gauge("oncoscan_feature_mean", "Moyenne de la variable", ["feature", "dataset"])
MALIGNANT_SHARE = Gauge("oncoscan_malignant_share", "Part de M", ["dataset"])
LIVE_ACCURACY = Gauge("oncoscan_live_accuracy", "Accuracy sur les cas confirmés")
LIVE_RECALL = Gauge("oncoscan_live_recall", "Recall (malins détectés) sur les cas confirmés")
LIVE_PRECISION = Gauge("oncoscan_live_precision", "Précision sur les cas confirmés")
LIVE_FN = Gauge("oncoscan_live_false_negatives", "Malins confirmés prédits bénins")
LIVE_FP = Gauge("oncoscan_live_false_positives", "Bénins confirmés prédits malins")

NAN = float("nan")

def load_reference() -> pd.DataFrame:
    df = pd.read_csv(REFERENCE_PATH)
    missing = set(FEATURES + ["diagnosis"]) - set(df.columns)
    if missing:
        raise RuntimeError(f"Colonnes manquantes dans {REFERENCE_PATH}: {sorted(missing)}")
    return df

def load_current() -> pd.DataFrame:
    query = f"""
        SELECT {", ".join(FEATURES)}, prediction, probability_malignant,
               model_version, doctor_feedback, created_at
        FROM predictions
        WHERE created_at >= now() - make_interval(days => %s)
    """
    with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
        cur.execute(query, (WINDOW_DAYS,))
        columns = [c.name for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)

def run_drift(reference: pd.DataFrame, current: pd.DataFrame) -> Path:
    """Rapport Evidently : drift des 5 variables + drift de la classe prédite."""
    ref = reference[FEATURES].astype(float).assign(prediction=reference["diagnosis"].astype(str))
    cur = current[FEATURES].astype(float).assign(prediction=current["prediction"].astype(str))

    definition = DataDefinition(numerical_columns=FEATURES, categorical_columns=["prediction"])
    report = Report([DataDriftPreset()])
    snapshot = report.run(
        Dataset.from_pandas(cur, data_definition=definition),
        Dataset.from_pandas(ref, data_definition=definition))

    export_drift_values(snapshot.dict())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"drift_{stamp}.html"
    snapshot.save_html(str(path))
    (REPORTS_DIR / "latest.html").write_bytes(path.read_bytes())

    for old in sorted(REPORTS_DIR.glob("drift_*.html"))[:-KEEP_REPORTS]:
        old.unlink(missing_ok=True)
    return path

def export_drift_values(result: dict) -> None:
    """Extrait les valeurs du snapshot Evidently vers les gauges Prometheus.

    DataDriftPreset produit :
    - DriftedColumnsCount  -> {"count": n, "share": x}
    - ValueDrift(column=c) -> score de drift de la colonne (p-value ou distance),
      avec le seuil dans sa config ; « drift » = score < seuil (tests statistiques)
      ou score >= seuil (distances).
    """
    drifted = 0
    for metric in result.get("metrics", []):
        config = metric.get("config", {})
        value = metric.get("value")
        kind = str(config.get("type", ""))

        if kind.endswith("DriftedColumnsCount") and isinstance(value, dict):
            DRIFT_COUNT.set(float(value.get("count", NAN)))
            DRIFT_SHARE.set(float(value.get("share", NAN)))

        elif kind.endswith("ValueDrift"):
            column = config.get("column")
            if column is None or not isinstance(value, (int, float)):
                continue
            method = str(config.get("method", "")).lower()
            threshold = config.get("threshold")
            if threshold is None:
                is_drift = NAN
            elif "p_value" in method or "test" in method:
                is_drift = float(value < threshold)
            else:
                is_drift = float(value >= threshold)
            label = column if column in FEATURES else "prediction"
            FEATURE_DRIFT.labels(feature=label).set(float(value))
            FEATURE_DRIFTED.labels(feature=label).set(is_drift)
            drifted += int(is_drift == 1.0)
    logger.info("Drift : %s colonne(s) en drift", drifted)

def export_descriptive(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    for feature in FEATURES:
        FEATURE_MEAN.labels(feature=feature, dataset="reference").set(float(reference[feature].mean()))
        FEATURE_MEAN.labels(feature=feature, dataset="production").set(
            float(current[feature].mean()) if len(current) else NAN)
    MALIGNANT_SHARE.labels(dataset="reference").set(float((reference["diagnosis"] == "M").mean()))
    MALIGNANT_SHARE.labels(dataset="production").set(
        float((current["prediction"] == "M").mean()) if len(current) else NAN)

def export_performance(current: pd.DataFrame) -> None:
    """Performance réelle : prédiction du modèle vs diagnostic confirmé par le médecin."""
    confirmed = current[current["doctor_feedback"].isin(["M", "B"])]
    CONFIRMED_ROWS.set(len(confirmed))
    if confirmed.empty:
        for gauge in (LIVE_ACCURACY, LIVE_RECALL, LIVE_PRECISION, LIVE_FN, LIVE_FP):
            gauge.set(NAN)
        return

    pred = confirmed["prediction"].str.strip()
    true = confirmed["doctor_feedback"]
    tp = int(((pred == "M") & (true == "M")).sum())
    fp = int(((pred == "M") & (true == "B")).sum())
    fn = int(((pred == "B") & (true == "M")).sum())

    LIVE_ACCURACY.set(float((pred == true).mean()))
    LIVE_RECALL.set(tp / (tp + fn) if (tp + fn) else NAN)
    LIVE_PRECISION.set(tp / (tp + fp) if (tp + fp) else NAN)
    LIVE_FN.set(fn)
    LIVE_FP.set(fp)


def clear_drift() -> None:
    DRIFT_SHARE.set(NAN)
    DRIFT_COUNT.set(NAN)
    for feature in FEATURES + ["prediction"]:
        FEATURE_DRIFT.labels(feature=feature).set(NAN)
        FEATURE_DRIFTED.labels(feature=feature).set(NAN)

def run_once(reference: pd.DataFrame) -> None:
    current = load_current()
    if not current.empty:
        current["prediction"] = current["prediction"].str.strip()
    CURRENT_ROWS.set(len(current))
    export_descriptive(reference, current)
    export_performance(current)

    if len(current) < MIN_ROWS:
        clear_drift()
        logger.info("%s prédiction(s) sur %s jours (< %s) : drift non calculé.",
                    len(current), WINDOW_DAYS, MIN_ROWS)
    else:
        path = run_drift(reference, current)
        logger.info("Rapport généré : %s (%s prédictions)", path.name, len(current))

def loop() -> None:
    reference = load_reference()
    logger.info("Référence : %s lignes (%s)", len(reference), REFERENCE_PATH)
    while True:
        try:
            run_once(reference)
            LAST_RUN_OK.set(1)
        except Exception:
            logger.exception("Échec du calcul de monitoring")
            LAST_RUN_OK.set(0)
        LAST_RUN_TS.set(time.time())
        time.sleep(INTERVAL_SECONDS)

# Serveur HTTP : /metrics pour Prometheus et rapports HTML 
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            self._send(200, generate_latest(), CONTENT_TYPE_LATEST)
        elif self.path in ("/", "/reports", "/reports/"):
            self._send(200, self._index().encode(), "text/html; charset=utf-8")
        elif self.path.startswith("/reports/"):
            name = self.path.removeprefix("/reports/")
            file = REPORTS_DIR / name
            # Seuls les fichiers .html du dossier, sans remontée de chemin
            if "/" in name or ".." in name or not name.endswith(".html") or not file.is_file():
                self._send(404, b"Not found", "text/plain")
            else:
                self._send(200, file.read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, b"Not found", "text/plain")

    def _index(self) -> str:
        files = sorted(REPORTS_DIR.glob("drift_*.html"), reverse=True)
        items = "".join(
            f'<li><a href="/reports/{html.escape(f.name)}">{html.escape(f.name)}</a></li>' for f in files)
        latest = '<p><a href="/reports/latest.html">Dernier rapport</a></p>' if files else (
            f"<p>Aucun rapport pour l'instant (il faut au moins {MIN_ROWS} prédictions "
            f"sur les {WINDOW_DAYS} derniers jours).</p>")
        return (f"<!doctype html><meta charset='utf-8'><title>OncoScan - Evidently</title>"
                f"<h1>Rapports de drift OncoScan</h1>{latest}<ul>{items}</ul>")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # pas de log pour chaque scrape Prometheus
        pass
    
if __name__ == "__main__":
    threading.Thread(target=loop, daemon=True).start()
    logger.info("Serveur de monitoring sur le port %s", PORT)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
