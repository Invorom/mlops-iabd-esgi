"""Entrainement du modele de classification (baseline).

Seance 5 - TP MLflow Tracking
    Ce script entraine et evalue un modele SANS aucun suivi d'experience.
    Votre mission : instrumenter cet entrainement avec MLflow (voir les TODO).
    La baseline fonctionne deja : `python -m mlproject.train` doit s'executer
    tel quel une fois config.py adapte a votre dataset (TP S0).
"""
from __future__ import annotations

import argparse

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import MODEL_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, MODEL_NAME
from src.data import load_data, split
from src.features import build_preprocessor

# (S5-1) : importer mlflow et mlflow.sklearn
import mlflow
import mlflow.sklearn


def build_model(c: float = 1.0, max_iter: int = 1000) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", LogisticRegression(C=c, max_iter=max_iter)),
        ]
    )


def train(c: float = 1.0, max_iter: int = 1000) -> dict:
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    # (S5-2) : configurer l'URI de tracking (mlflow.set_tracking_uri) et l'experience
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # (S5-3) : ouvrir un run englobant l'entrainement et l'evaluation (with mlflow.start_run())
    with mlflow.start_run():
        model = build_model(c=c, max_iter=max_iter)
        model.fit(x_train, y_train)

        proba = model.predict_proba(x_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics = {
            "f1": float(f1_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }
        print(f"f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")

        # (S5-4) : logger les parametres (c, max_iter) avec mlflow.log_params
        mlflow.log_params({
            "c": c,
            "max_iter": max_iter,
        })

        # (S5-5) : logger les metriques (f1, roc_auc) avec mlflow.log_metrics
        mlflow.log_metrics(metrics)

        # (S5-6) : logger le modele avec mlflow.sklearn.log_model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        # (S5-7 bonus) : sauvegarder la matrice de confusion en image et la logger en artefact
        try:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            import matplotlib.pyplot as plt
            import os
            import tempfile

            cm = confusion_matrix(y_test, preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Légitime", "Fraude"])
            disp.plot(cmap="Blues")
            plt.title("Confusion Matrix")
            
            temp_dir = tempfile.gettempdir()
            plot_path = os.path.join(temp_dir, "confusion_matrix.png")
            plt.savefig(plot_path)
            plt.close()
            
            mlflow.log_artifact(plot_path)
            # Nettoyage du fichier local
            if os.path.exists(plot_path):
                os.remove(plot_path)
        except Exception as e:
            print(f"Erreur lors de la génération de la matrice de confusion : {e}")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / "model.joblib")
        
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
