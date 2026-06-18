"""DAG Airflow - pipeline de re-entrainement du modele.

Seance 17 - TP Airflow
    Pipeline : preparation des donnees -> entrainement -> controle qualite.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# f1 minimal du modele entraine pour que le pipeline soit considere comme reussi.
QUALITY_THRESHOLD = 0.65

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_prepare_data(**context) -> None:
    """Charge et verifie que le jeu de donnees est disponible."""
    from src.data import load_data

    df = load_data()
    logger.info("Donnees chargees : %d lignes, %d colonnes", df.shape[0], df.shape[1])


def task_train(**context) -> None:
    """Entraine le modele et pousse le f1 dans XCom."""
    from src.train import train

    metrics = train()
    context["ti"].xcom_push(key="f1", value=metrics["f1"])
    logger.info("Entrainement termine : f1=%.3f, roc_auc=%.3f", metrics["f1"], metrics["roc_auc"])


def task_check_quality(**context) -> None:
    """Verifie que le f1 depasse le seuil de qualite."""
    f1 = context["ti"].xcom_pull(task_ids="train", key="f1")
    if f1 is None:
        raise ValueError("Impossible de recuperer le f1 depuis XCom")
    if f1 < QUALITY_THRESHOLD:
        raise ValueError(
            f"Qualite insuffisante : f1={f1:.3f} < seuil={QUALITY_THRESHOLD}"
        )
    logger.info("Controle qualite OK : f1=%.3f >= %.2f", f1, QUALITY_THRESHOLD)


with DAG(
    dag_id="model_retraining",
    description="Prepare les donnees, reentraine le modele et controle sa qualite",
    schedule="0 3 * * 1",  # Tous les lundis a 3h
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["classification", "training"],
) as dag:
    prepare = PythonOperator(task_id="prepare_data", python_callable=task_prepare_data)
    train_task = PythonOperator(task_id="train", python_callable=task_train)
    check = PythonOperator(task_id="check_quality", python_callable=task_check_quality)

    prepare >> train_task >> check
