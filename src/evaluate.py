"""Evaluation automatisee et validation du modele (squelette).

Seance 11 - TP Tests Donnees & Modele
    `mlflow.models.evaluate` calcule en une passe un ensemble de metriques et
    d'artefacts (matrice de confusion, courbes ROC / precision-rappel) sur un
    jeu d'evaluation. `mlflow.validate_evaluation_results` applique ensuite une
    porte qualite : le modele est rejete (exception) si une metrique passe sous
    son seuil. Completez les TODO (S11-1, S11-2, S11-3).
    Pre-requis : un modele enregistre au Model Registry (Seances 5-6).

Le jeu d'evaluation est logue comme dataset MLflow (tracabilite).

Lancement :
    python -m mlproject.evaluate                       # derniere version du registry
    python -m mlproject.evaluate --model-uri models:/classifier/1
    python -m mlproject.evaluate --no-validate         # evalue sans porte qualite
"""
from __future__ import annotations

import argparse
import logging

import mlflow
import mlflow.data
import mlflow.models
from mlflow.exceptions import MlflowException
from mlflow.models import MetricThreshold

from src.config import (
    DATA_PATH,
    EVAL_F1_MIN,
    EVAL_ROC_AUC_MIN,
    MODEL_DIR,
    MODEL_NAME,
    TARGET,
)
from src.data import load_data, split
from src.tracking import setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def latest_model_uri() -> str:
    """Resoudre l'URI de la derniere version enregistree de ``MODEL_NAME`` [FOURNI].

    Returns
    -------
    str
        URI MLflow de la forme ``models:/<MODEL_NAME>/<version>``.
    """
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            f"Aucune version enregistree pour '{MODEL_NAME}'. "
            "Lancez d'abord un entrainement (make train)."
        )
    latest = max(versions, key=lambda v: int(v.version))
    return f"models:/{MODEL_NAME}/{latest.version}"


def build_thresholds() -> dict[str, MetricThreshold]:
    """Construire les seuils de validation a partir de la configuration.

    A implementer : retourner un dictionnaire {nom_metrique: MetricThreshold}.
    Les noms de metriques produits par l'evaluateur "classifier" sont
    notamment `roc_auc` et `f1_score`. Exemple de seuil :
        MetricThreshold(threshold=EVAL_ROC_AUC_MIN, greater_is_better=True)

    Returns
    -------
    dict of str to MetricThreshold
        Seuils minimaux sur ``roc_auc`` et ``f1_score``.
    """
    return {
        "roc_auc": MetricThreshold(threshold=EVAL_ROC_AUC_MIN, greater_is_better=True),
        "f1_score": MetricThreshold(threshold=EVAL_F1_MIN, greater_is_better=True),
    }


def evaluate_model(model_uri: str | None = None, validate: bool = True):
    """Evaluer un modele du registry et, optionnellement, valider les seuils.

    Parameters
    ----------
    model_uri : str, optional
        URI MLflow du modele a evaluer. Par defaut, la derniere version
        enregistree de ``MODEL_NAME`` (via ``latest_model_uri``).
    validate : bool, optional
        Applique la porte qualite, par defaut True. Leve une exception si un
        seuil n'est pas atteint.

    Returns
    -------
    mlflow.models.EvaluationResult
        Resultat de l'evaluation (metriques et artefacts).
    """
    df = load_data()
    _, x_test, _, y_test = split(df)

    setup_experiment()
    logger.info("Evaluation du modele...")

    with mlflow.start_run(run_name="evaluate"):
        # Charger le modele local (rapide) au lieu de telecharger depuis le registry
        import joblib
        from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        model_path = MODEL_DIR / "model.joblib"
        if not model_path.exists():
            raise RuntimeError(f"Modele introuvable : {model_path}. Lancez d'abord un entrainement.")
        
        model = joblib.load(model_path)
        logger.info("Modele charge depuis %s", model_path)

        # Predictions
        preds = model.predict(x_test)
        proba = model.predict_proba(x_test)[:, 1]

        # Metriques
        f1 = float(f1_score(y_test, preds))
        roc_auc = float(roc_auc_score(y_test, proba))
        
        logger.info("f1_score=%.3f roc_auc=%.3f", f1, roc_auc)

        # Logger dans MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Matrice de confusion
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title("Matrice de confusion - Evaluation")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # Rapport de classification
        report = classification_report(y_test, preds)
        mlflow.log_text(report, "classification_report.txt")

        # Tracabilite du dataset
        eval_df = x_test.copy()
        eval_df[TARGET] = y_test.values
        dataset = mlflow.data.from_pandas(eval_df, source=str(DATA_PATH), targets=TARGET, name="eval")  # type: ignore[attr-defined]
        mlflow.log_input(dataset, context="evaluation")

        # Porte qualite
        if validate:
            thresholds = build_thresholds()
            metrics = {"roc_auc": roc_auc, "f1_score": f1}
            for metric_name, threshold in thresholds.items():
                value = metrics.get(metric_name, 0.0)
                if value < threshold.threshold:
                    raise MlflowException(
                        f"Validation echouee : {metric_name}={value:.3f} < seuil={threshold.threshold}"
                    )
            logger.info("Validation reussie : tous les seuils sont respectes.")

        return {"f1_score": f1, "roc_auc": roc_auc}


def main() -> None:
    """Point d'entree en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-uri",
        default=None,
        help="URI du modele a evaluer (defaut: derniere version de MODEL_NAME)",
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Evalue sans appliquer la porte qualite (seuils)",
    )
    args = parser.parse_args()

    model_uri = args.model_uri or None
    try:
        evaluate_model(model_uri=model_uri, validate=args.validate)
    except MlflowException as exc:
        logger.error("Validation echouee : %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
