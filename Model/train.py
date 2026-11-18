"""Entraînement du modèle OncoScan.

Démarche, dans l'ordre :
    1. comparaison de 6 modèles en validation croisée (sur le train uniquement) ;
    2. choix de la calibration des probabilités ;
    3. choix du seuil de décision (meilleur recall à précision >= 90 %) ;
    4. entraînement final puis une seule évaluation sur le jeu de test ;
    5. sauvegarde du modèle + fiche JSON, figures, et suivi de l'expérience dans MLflow.

Utilisation :
    python Model/train.py                    # entraîne, sauvegarde, trace dans MLflow
    python Model/train.py --no-mlflow        # sans MLflow
    python Model/train.py --output-dir tmp/  # sauvegarde ailleurs (pour essayer sans écraser)
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # consulter les expériences
"""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : le script tourne aussi bien en local qu'en CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    FixedThresholdClassifier,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Paramètres
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "Data" / "process" / "cancer_select.csv"
MODELS_DIR = BASE_DIR / "Save_models"
FIGURES_DIR = BASE_DIR / "Image_models"

SELECTED_FEATURES = ["perimeter", "concave_points", "texture", "smoothness", "symmetry"]

# Le découpage train/test doit être identique dans Tests/test_model.py,
# sinon les tests évalueraient le modèle sur des données qu'il a déjà vues.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Mêmes seuils que Tests/test_model.py
MIN_PRECISION = 0.90  # contrainte utilisée pour choisir le modèle et le seuil
MIN_RECALL = 0.95     # seulement contrôlé et signalé

# Nom de fichier fixe : l'API, le Dockerfile et la CI n'ont pas à changer
# quand un autre algorithme gagne la comparaison.
MODEL_FILENAME = "oncoscan_pipeline.joblib"

# MLflow : base SQLite locale par défaut, ou serveur distant via MLFLOW_TRACKING_URI
MLFLOW_EXPERIMENT = "oncoscan"
MLFLOW_REGISTERED_MODEL = "oncoscan"
DEFAULT_TRACKING_URI = f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}"


# Modèles candidats
def build_candidates(y_train: np.ndarray) -> dict[str, Pipeline]:
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    base_models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=4,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=RANDOM_STATE,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, class_weight="balanced"),
            n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE,
        ),
        "Naive Bayes": GaussianNB(),
    }
    return {
        name: Pipeline([("scaler", MinMaxScaler()), ("model", model)])
        for name, model in base_models.items()
    }


# 1) Choix du modèle
def select_model(pipelines, X_train, y_train, cv):
    """Compare les modèles en validation croisée, sur le train uniquement.

    Le jeu de test reste intact jusqu'à l'évaluation finale : s'il servait aussi
    à choisir le modèle, les scores annoncés seraient trop optimistes.
    Le recall seul ne suffit pas non plus (un modèle qui répond « maligne » à
    tout le monde a 100 % de recall), d'où la contrainte de précision.
    """
    rows = []
    for name, pipeline in pipelines.items():
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, n_jobs=-1,
            scoring=["recall", "precision", "f1", "roc_auc"],
        )
        rows.append({
            "Model": name,
            "Recall": scores["test_recall"].mean(),
            "Recall_std": scores["test_recall"].std(),
            "Precision": scores["test_precision"].mean(),
            "F1": scores["test_f1"].mean(),
            "AUC": scores["test_roc_auc"].mean(),
        })
        print(f"{name:<20} | Recall {rows[-1]['Recall']:.3f} | Précision {rows[-1]['Precision']:.3f} "
              f"| F1 {rows[-1]['F1']:.3f} | AUC {rows[-1]['AUC']:.3f}")

    results_df = pd.DataFrame(rows)

    # Meilleur recall parmi les modèles assez précis ; à égalité, la meilleure AUC
    eligible = results_df[results_df["Precision"] >= MIN_PRECISION]
    if eligible.empty:
        print(f"/!\\ Aucun modèle n'atteint {MIN_PRECISION:.0%} de précision : choix sur le F1.")
        best = results_df.sort_values(["F1", "AUC"], ascending=False).iloc[0]
    else:
        best = eligible.sort_values(["Recall", "AUC"], ascending=False).iloc[0]

    return results_df.sort_values("Recall"), best["Model"]


# 2) Calibration
def select_calibration(pipeline, X_train, y_train, cv):
    """Compare « sans calibration » et « sigmoid » sur des prédictions hors échantillon.

    Pas d'isotonic : avec environ 450 lignes d'entraînement elle sur-apprend et
    sort des probabilités à 0 ou 100 % (essayé : log loss presque triplée).
    """
    candidates = {
        "aucune": clone(pipeline),
        "sigmoid": CalibratedClassifierCV(clone(pipeline), method="sigmoid", cv=5),
    }
    scores = {}
    oof_probas = {}
    for method, estimator in candidates.items():
        proba = cross_val_predict(
            estimator, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        oof_probas[method] = proba
        scores[method] = {
            "brier": brier_score_loss(y_train, proba),
            "log_loss": log_loss(y_train, proba),
        }
        print(f"Calibration {method:<9} | Brier {scores[method]['brier']:.4f} "
              f"| Log loss {scores[method]['log_loss']:.4f}")

    # On ne calibre que si les deux scores s'améliorent : un modèle déjà bien
    # calibré (une Random Forest par exemple) reste tel quel.
    improves = (
        scores["sigmoid"]["brier"] < scores["aucune"]["brier"]
        and scores["sigmoid"]["log_loss"] < scores["aucune"]["log_loss"]
    )
    best_method = "sigmoid" if improves else "aucune"
    return best_method, candidates[best_method], oof_probas[best_method], scores


# 3) Seuil de décision
def select_threshold(y_true, proba, min_precision=MIN_PRECISION):
    """Seuil qui donne le meilleur recall en gardant une précision >= min_precision.

    Le 0.5 par défaut n'a aucun sens médical : on préfère rater le moins de
    cancers possible, sans pour autant noyer les médecins sous les fausses alertes.
    Le calcul se fait sur des prédictions hors échantillon du train, jamais sur le test.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    ok = precision >= min_precision
    if not ok.any():
        print(f"/!\\ Précision de {min_precision:.0%} inatteignable : seuil 0.5 conservé.")
        return 0.5
    best_recall = recall[ok].max()
    # À recall égal, on prend le seuil le plus haut (moins de fausses alertes)
    return float(thresholds[ok & (recall == best_recall)].max())


# Figures
def build_figures(results_df, cm, y_test, proba_raw, proba_final, best_pipeline,
                  best_model_name, calib_method, X_train, y_train, cv):
    figures = {}

    # Comparaison des modèles, avec l'écart-type entre les plis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        results_df["Model"], results_df["Recall"], xerr=results_df["Recall_std"],
        color=plt.cm.RdYlGn(results_df["Recall"]), edgecolor="black", capsize=4,
    )
    for i, recall in enumerate(results_df["Recall"]):
        ax.text(recall + 0.03, i, f"{recall:.3f}", va="center", fontweight="bold")
    ax.set_xlabel("Recall moyen (validation croisée)", fontsize=12, fontweight="bold")
    ax.set_title("Comparaison des modèles (CV 5 plis sur le train)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout()
    figures["Comparaison_models.png"] = fig

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Bénin (0)", "Malin (1)"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Matrice de confusion - {best_model_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    figures["Matrice_confusion.png"] = fig

    # Plus la courbe suit la diagonale, plus les pourcentages affichés sont fiables
    fig, ax = plt.subplots(figsize=(7, 6))
    CalibrationDisplay.from_predictions(
        y_test, proba_raw, n_bins=8, ax=ax, name=f"{best_model_name} (non calibré)"
    )
    CalibrationDisplay.from_predictions(
        y_test, proba_final, n_bins=8, ax=ax, name=f"Calibré : {calib_method}"
    )
    ax.set_title("Courbe de calibration (test)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    figures["Courbe_calibration.png"] = fig

    print("Génération de la courbe d'apprentissage...")
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=clone(best_pipeline), X=X_train, y=y_train, cv=cv,
        scoring="recall", n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Score entraînement")
    ax.plot(train_sizes, val_mean, "o-", color="red", label="Score validation")
    ax.set_xlabel("Nombre d'échantillons d'entraînement", fontweight="bold")
    ax.set_ylabel("Recall", fontweight="bold")
    ax.set_title(f"Courbe d'apprentissage - {best_model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    figures["Courbe_apprentissage.png"] = fig

    gap = train_mean[-1] - val_mean[-1]
    if gap > 0.1:
        print(f"/!\\ Écart train/validation = {gap:.3f} : le modèle sur-apprend peut-être.\n")
    else:
        print(f"OK : écart train/validation = {gap:.3f}, pas de sur-apprentissage visible.\n")

    return figures


# Entraînement complet
def train_model(
    df: pd.DataFrame,
    target_col: str = "diagnosis",
    selected_features: list[str] | None = None,
    save_models: bool = True,
    output_dir: Path = MODELS_DIR,
):
    """Entraîne, évalue et (si demandé) sauvegarde le modèle.

    Renvoie un dictionnaire avec tout ce qu'il faut pour tracer l'expérience :
    modèle final, métriques, paramètres retenus, figures, fiche de métadonnées.
    """
    selected_features = selected_features or SELECTED_FEATURES

    # Si une colonne manque, on s'arrête : un modèle entraîné sur moins de variables
    # que celles envoyées par l'API planterait en production.
    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes introuvables : {missing}")

    X = df[selected_features]
    y = df[target_col]
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
        y = y.map({"M": 1, "B": 0}).astype(int)

    print("Distribution des classes :")
    print(f"Bénin (B=0) : {(y == 0).sum()} ({(y == 0).mean() * 100:.1f}%)")
    print(f"Malin (M=1) : {(y == 1).sum()} ({(y == 1).mean() * 100:.1f}%)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    print(f"Train : {X_train.shape[0]} échantillons | Test : {X_test.shape[0]} échantillons\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipelines = build_candidates(y_train)

    print("1) Comparaison des modèles (validation croisée sur le train)")
    print("-" * 80)
    results_df, best_model_name = select_model(pipelines, X_train, y_train, cv)
    print("-" * 80)
    print(f"Modèle retenu : {best_model_name}\n")
    best_pipeline = pipelines[best_model_name]

    print("2) Choix de la calibration (validation croisée sur le train)")
    print("-" * 80)
    calib_method, calibrated_estimator, oof_proba, calib_scores = select_calibration(
        best_pipeline, X_train, y_train, cv
    )
    print(f"Calibration retenue : {calib_method}\n")

    threshold = select_threshold(y_train, oof_proba)
    print(f"3) Seuil de décision retenu : {threshold:.3f} "
          f"(meilleur recall avec une précision >= {MIN_PRECISION:.0%})\n")

    # FixedThresholdClassifier garde predict_proba et classes_ : l'API et les
    # tests n'ont pas à savoir qu'il y a un seuil personnalisé derrière.
    final_model = FixedThresholdClassifier(
        calibrated_estimator, threshold=threshold, response_method="predict_proba"
    )
    final_model.fit(X_train, y_train)

    # Modèle brut (sans calibration ni seuil), uniquement pour mesurer ce que la calibration apporte
    raw_model = clone(best_pipeline).fit(X_train, y_train)

    print(f"{'=' * 60}\n4) ÉVALUATION SUR LE TEST - {best_model_name}\n{'=' * 60}\n")
    y_pred = final_model.predict(X_test)
    proba_final = final_model.predict_proba(X_test)[:, 1]
    proba_raw = raw_model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, proba_final),
        "brier_avant": brier_score_loss(y_test, proba_raw),
        "brier_apres": brier_score_loss(y_test, proba_final),
        "log_loss_avant": log_loss(y_test, proba_raw),
        "log_loss_apres": log_loss(y_test, proba_final),
    }

    report = classification_report(
        y_test, y_pred, target_names=["Bénin (0)", "Malin (1)"], digits=3, zero_division=0
    )
    print(report)
    print(f"AUC         : {test_metrics['auc']:.3f}")
    print(f"Brier score : {test_metrics['brier_avant']:.4f} -> {test_metrics['brier_apres']:.4f}")
    print(f"Log loss    : {test_metrics['log_loss_avant']:.4f} -> {test_metrics['log_loss_apres']:.4f}\n")

    # Prévient tout de suite si la CI (Tests/test_model.py) va refuser ce modèle
    if test_metrics["recall"] < MIN_RECALL or test_metrics["precision"] < MIN_PRECISION:
        print(f"/!\\ Seuils des tests non atteints (recall >= {MIN_RECALL}, précision >= {MIN_PRECISION}) : "
              "la CI refusera ce modèle.\n")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN = {tn}  bénins bien classés")
    print(f"FP = {fp}  bénins classés malins (fausses alertes)")
    print(f"FN = {fn}  malins manqués, l'erreur à éviter")
    print(f"TP = {tp}  malins bien détectés\n")

    figures = build_figures(
        results_df, cm, y_test, proba_raw, proba_final, best_pipeline,
        best_model_name, calib_method, X_train, y_train, cv,
    )

    # Fiche qui accompagne le modèle : ce qu'il est, avec quoi il a été fait, ce qu'il vaut
    metadata = {
        "model": best_model_name,
        "calibration": calib_method,
        "decision_threshold": round(threshold, 4),
        "min_precision": MIN_PRECISION,
        "features": selected_features,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "calibration_cv": {
            m: {k: round(float(v), 4) for k, v in s.items()} for m, s in calib_scores.items()
        },
    }

    model_path = None
    if save_models:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / MODEL_FILENAME
        joblib.dump(final_model, model_path)
        model_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Modèle sauvegardé : {model_path}")
        print(f"Fiche du modèle   : {model_path.with_suffix('.json')}")

    return {
        "model": final_model,
        "model_path": model_path,
        "metadata": metadata,
        "cv_results": results_df,
        "report": report,
        "figures": figures,
        "X_test": X_test,
    }


# Suivi MLflow
def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=BASE_DIR, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "inconnu"


def trusted_model_types(model) -> list[str]:
    """Types internes du modèle que MLflow (format skops) doit accepter de recharger.

    skops refuse par défaut les classes qu'il ne connaît pas, comme celles de la
    calibration. On n'autorise que ce qui vient de scikit-learn ou de XGBoost :
    c'est notre propre modèle, il n'y a donc rien d'autre à faire confiance.
    """
    import skops.io

    untrusted = skops.io.get_untrusted_types(data=skops.io.dumps(model))
    return [name for name in untrusted if name.startswith(("sklearn.", "xgboost."))]


def log_to_mlflow(outputs: dict, data_path: Path, df: pd.DataFrame) -> None:
    """Trace l'expérience : paramètres, métriques, figures, données et modèle.

    Le modèle est aussi inscrit au registre MLflow (« oncoscan ») : chaque
    entraînement crée une nouvelle version, ce qui permet de comparer et de revenir
    en arrière. Le fichier .joblib dans Save_models/ reste la source utilisée par l'API.
    """
    import mlflow  # import local : MLflow n'est requis que pour l'entraînement
    import mlflow.sklearn
    from mlflow.models import infer_signature

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)

    # Avec la base SQLite locale, les artefacts vont dans ./mlruns (ignoré par git)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        artifact_location = (BASE_DIR / "mlruns").as_uri() if tracking_uri.startswith("sqlite") else None
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=artifact_location)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    meta = outputs["metadata"]
    with mlflow.start_run(run_name=f"{meta['model']} - {meta['trained_at']}"):
        mlflow.set_tags({
            "git_commit": git_commit(),
            "data_sha256": file_sha256(data_path),
            "sklearn_version": meta["sklearn_version"],
        })
        mlflow.log_params({
            "model": meta["model"],
            "calibration": meta["calibration"],
            "decision_threshold": meta["decision_threshold"],
            "min_precision": meta["min_precision"],
            "features": ",".join(meta["features"]),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_train": meta["n_train"],
            "n_test": meta["n_test"],
        })
        mlflow.log_metrics({f"test_{k}": v for k, v in meta["test_metrics"].items()})

        # Scores de validation croisée de chaque candidat, pour garder la trace de la comparaison
        for row in outputs["cv_results"].itertuples():
            prefix = "cv_" + row.Model.lower().replace(" ", "_")
            mlflow.log_metrics({
                f"{prefix}_recall": row.Recall,
                f"{prefix}_precision": row.Precision,
                f"{prefix}_auc": row.AUC,
            })

        mlflow.log_input(
            mlflow.data.from_pandas(df, source=str(data_path), name=data_path.name, targets="diagnosis"),
            context="training",
        )
        for name, fig in outputs["figures"].items():
            mlflow.log_figure(fig, f"figures/{name}")
        mlflow.log_text(outputs["report"], "classification_report.txt")
        mlflow.log_dict(meta, "model_card.json")

        X_example = outputs["X_test"].head(5)
        mlflow.sklearn.log_model(
            outputs["model"],
            name="model",
            signature=infer_signature(X_example, outputs["model"].predict(X_example)),
            input_example=X_example,
            registered_model_name=MLFLOW_REGISTERED_MODEL,
            skops_trusted_types=trusted_model_types(outputs["model"]),
        )
        print(f"Expérience tracée dans MLflow ({tracking_uri})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîne le modèle OncoScan.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="CSV d'entraînement")
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR,
                        help="Dossier du modèle et de sa fiche (défaut : Save_models/)")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR,
                        help="Dossier des figures (défaut : Image_models/)")
    parser.add_argument("--no-mlflow", action="store_true", help="Ne pas tracer dans MLflow")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = pd.read_csv(args.data)
    outputs = train_model(data, output_dir=args.output_dir)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    for filename, figure in outputs["figures"].items():
        figure.savefig(args.figures_dir / filename, dpi=120)
    print(f"Figures enregistrées dans {args.figures_dir}")

    if not args.no_mlflow:
        log_to_mlflow(outputs, args.data.resolve(), data)