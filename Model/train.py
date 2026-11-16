#=======================#
#   Import librairies   #
#=======================#
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
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


#==================#
#   Paramètres     #
#==================#
# Racine du projet (train.py est dans Model/) : les chemins ne dépendent plus
# du dossier depuis lequel on lance le script.
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "Data" / "process" / "cancer_select.csv"
MODELS_DIR = BASE_DIR / "Save_models"

SELECTED_FEATURES = ["perimeter", "concave_points", "texture", "smoothness", "symmetry"]

# Doivent rester identiques à ceux de Tests/test_model.py
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Mêmes seuils que Tests/test_model.py
MIN_PRECISION = 0.90  # contrainte pour choisir le modèle et le seuil de décision
MIN_RECALL = 0.95     # seulement vérifié et signalé

# Nom fixe : l'API, le Dockerfile et la CI n'ont pas à changer quand le meilleur modèle change.
# (avant : "{nom_du_modele}_pipeline.joblib", qui cassait tout si XGBoost gagnait)
MODEL_FILENAME = "oncoscan_pipeline.joblib"


#==========================#
#   Modèles candidats      #
#==========================#
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
        # Était importé mais jamais utilisé
        "Naive Bayes": GaussianNB(),
    }
    return {
        name: Pipeline([("scaler", MinMaxScaler()), ("model", model)])
        for name, model in base_models.items()
    }


#================================================#
#   1) Choix du modèle par validation croisée     #
#================================================#
def select_model(pipelines, X_train, y_train, cv):
    """Compare les modèles sur le train uniquement.

    Avant, le choix se faisait sur le jeu de test : il servait à la fois à choisir
    et à évaluer, donc les scores finaux étaient trop optimistes.
    Et le recall seul ne suffit pas : un modèle qui dit "maligne" à tout le monde a 100 %.
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


#===========================================#
#   2) Choix de la méthode de calibration    #
#===========================================#
def select_calibration(pipeline, X_train, y_train, cv):
    """Compare sans calibration / sigmoid sur des prédictions hors échantillon (train).

    Pas d'isotonic : avec ~450 lignes d'entraînement, elle sur-apprend et donne des
    probabilités à 0 ou 100 % (testé : log loss presque triplé sur le test).
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

    # On ne calibre que si ça améliore les deux scores (plus bas = probabilités plus fiables).
    # Un modèle déjà bien calibré (ex. Random Forest ici) reste tel quel.
    improves = (
        scores["sigmoid"]["brier"] < scores["aucune"]["brier"]
        and scores["sigmoid"]["log_loss"] < scores["aucune"]["log_loss"]
    )
    best_method = "sigmoid" if improves else "aucune"
    return best_method, candidates[best_method], oof_probas[best_method], scores


#==============================#
#   3) Seuil de décision        #
#==============================#
def select_threshold(y_true, proba, min_precision=MIN_PRECISION):
    """Seuil qui donne le meilleur recall tout en gardant une précision >= min_precision.

    Le seuil de 0.5 par défaut n'a rien de médical : on le règle pour rater le moins
    de cancers possible, sans noyer les médecins sous les fausses alertes.
    Calculé sur des prédictions hors échantillon du train, jamais sur le test.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    ok = precision >= min_precision
    if not ok.any():
        print(f"/!\\ Précision de {min_precision:.0%} inatteignable : seuil 0.5 conservé.")
        return 0.5
    best_recall = recall[ok].max()
    # À recall égal, le seuil le plus haut (moins de fausses alertes)
    return float(thresholds[ok & (recall == best_recall)].max())


#=============================#
#   Fonction d'entraînement   #
#=============================#
def train_model(
    df: pd.DataFrame,
    target_col: str = "diagnosis",
    selected_features: list[str] | None = None,
    save_models: bool = True,
    output_dir: Path = MODELS_DIR,
):
    selected_features = selected_features or SELECTED_FEATURES

    # Avant : on continuait avec les colonnes disponibles. Le modèle aurait alors
    # attendu moins de features que l'API n'en envoie -> erreur en prod.
    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes introuvables : {missing}")

    X = df[selected_features]
    y = df[target_col]
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
        y = y.map({"M": 1, "B": 0}).astype(int)

    print("Distribution des classes :")
    print(f"Bénin (B=0) : {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"Malin (M=1) : {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)\n")

    # Même découpage que dans les tests
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    print(f"Train set : {X_train.shape[0]} échantillons")
    print(f"Test set : {X_test.shape[0]} échantillons\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipelines = build_candidates(y_train)

    # 1) Choix du modèle
    print("1) Comparaison des modèles (validation croisée sur le train)")
    print("-" * 80)
    results_df, best_model_name = select_model(pipelines, X_train, y_train, cv)
    print("-" * 80)
    print(f"Modèle retenu : {best_model_name}\n")
    best_pipeline = pipelines[best_model_name]

    # 2) Calibration
    print("2) Choix de la calibration (validation croisée sur le train)")
    print("-" * 80)
    calib_method, calibrated_estimator, oof_proba, calib_scores = select_calibration(
        best_pipeline, X_train, y_train, cv
    )
    print(f"Calibration retenue : {calib_method}\n")

    # 3) Seuil de décision
    threshold = select_threshold(y_train, oof_proba)
    print(f"3) Seuil de décision retenu : {threshold:.3f} "
          f"(meilleur recall avec une précision ≥ {MIN_PRECISION:.0%})\n")

    # 4) Modèle final entraîné sur tout le train.
    # FixedThresholdClassifier garde predict_proba et classes_ : l'API et les tests marchent tels quels.
    final_model = FixedThresholdClassifier(
        calibrated_estimator, threshold=threshold, response_method="predict_proba"
    )
    final_model.fit(X_train, y_train)

    # Modèle de départ (sans calibration ni seuil), seulement pour comparer
    raw_model = clone(best_pipeline).fit(X_train, y_train)

    # 5) Évaluation unique sur le jeu de test
    print(f"{'='*60}\nÉVALUATION SUR LE TEST - {best_model_name}\n{'='*60}\n")
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

    rapport = classification_report(
        y_test, y_pred, target_names=["Bénin (0)", "Malin (1)"], digits=3, zero_division=0
    )
    print(rapport)
    print(f"AUC : {test_metrics['auc']:.3f}")
    print(f"Brier score : {test_metrics['brier_avant']:.4f} -> {test_metrics['brier_apres']:.4f}")
    print(f"Log loss    : {test_metrics['log_loss_avant']:.4f} -> {test_metrics['log_loss_apres']:.4f}\n")

    # Prévient tout de suite si la CI (Tests/test_model.py) va refuser ce modèle
    if test_metrics["recall"] < MIN_RECALL or test_metrics["precision"] < MIN_PRECISION:
        print(f"/!\\ Seuils des tests non atteints (recall ≥ {MIN_RECALL}, précision ≥ {MIN_PRECISION}) : "
              "la CI va échouer avec ce modèle.\n")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN = {tn}  Vrais Négatifs -- Bénins bien classés")
    print(f"FP = {fp}  Faux Positifs -- Bénins classés malins (fausse alarme)")
    print(f"FN = {fn}  Faux Négatifs -- Malins manqués *DANGEREUX*")
    print(f"TP = {tp}  Vrais Positifs -- Malins bien détectés\n")

    #======================#
    #   Graphiques         #
    #======================#
    # Comparaison des modèles (CV), avec l'écart-type
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
    plt.tight_layout()

    # Matrice de confusion
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Bénin (0)", "Malin (1)"]).plot(
        ax=ax_cm, colorbar=False, cmap="Blues"
    )
    ax_cm.set_title(f"Matrice de confusion -- {best_model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Courbe de calibration : plus la courbe suit la diagonale, plus les % sont fiables
    fig_calib, ax_calib = plt.subplots(figsize=(7, 6))
    CalibrationDisplay.from_predictions(
        y_test, proba_raw, n_bins=8, ax=ax_calib, name=f"{best_model_name} (avant)"
    )
    CalibrationDisplay.from_predictions(
        y_test, proba_final, n_bins=8, ax=ax_calib, name=f"Calibré : {calib_method}"
    )
    ax_calib.set_title("Courbe de calibration (test)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Courbe d'apprentissage du modèle retenu
    print("Génération de la courbe d'apprentissage...\n")
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=clone(best_pipeline), X=X_train, y=y_train, cv=cv,
        scoring="recall", n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    fig_learning, ax_learning = plt.subplots(figsize=(9, 5))
    ax_learning.plot(train_sizes, train_mean, "o-", color="blue", label="Score entraînement")
    ax_learning.plot(train_sizes, val_mean, "o-", color="red", label="Score validation")
    ax_learning.set_xlabel("Nombre d'échantillons d'entraînement", fontweight="bold")
    ax_learning.set_ylabel("Recall", fontweight="bold")
    ax_learning.set_title(f"Courbe d'apprentissage -- {best_model_name}", fontsize=13, fontweight="bold")
    ax_learning.legend(loc="lower right")
    ax_learning.grid(True, alpha=0.3, linestyle="--")
    ax_learning.set_ylim(0, 1.05)
    plt.tight_layout()

    gap = train_mean[-1] - val_mean[-1]
    if gap > 0.1:
        print(f"/!\\ Écart train/validation = {gap:.3f} : le modèle sur-apprend peut-être un peu.")
    else:
        print(f"OK : écart train/validation = {gap:.3f}, pas de sur-apprentissage visible.")

    #====================================================#
    #    Sauvegarde du modèle + fiche de métadonnées     #
    #====================================================#
    pipeline_filename = None
    if save_models:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        pipeline_filename = output_path / MODEL_FILENAME
        joblib.dump(final_model, pipeline_filename)

        # Fiche qui accompagne le modèle : ce qu'il est, avec quoi il a été fait, ce qu'il vaut
        metadata = {
            "model": best_model_name,
            "calibration": calib_method,
            "decision_threshold": round(threshold, 4),
            "min_precision": MIN_PRECISION,
            "features": selected_features,
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
            "calibration_cv": {
                m: {k: round(float(v), 4) for k, v in s.items()} for m, s in calib_scores.items()
            },
        }
        metadata_path = pipeline_filename.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\nModèle sauvegardé : {pipeline_filename}")
        print(f"Métadonnées       : {metadata_path}")

    return results_df, fig, rapport, fig_learning, fig_cm, fig_calib, pipeline_filename


#=========================#
#   Lancer l'entraînement #
#=========================#
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, sep=",")
    outputs = train_model(df, selected_features=SELECTED_FEATURES)
    plt.show()