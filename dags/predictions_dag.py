"""DAG Airflow - trafic de previsions quotidien.

Seance 17 - TP Airflow (suite)
    Planifie l'envoi quotidien d'un lot de previsions a l'API : chaque jour a
    10h, on echantillonne 20 lignes du jeu de donnees et on les envoie en
    POST /predict. Cela simule un flux de previsions en production.

Prerequis : l'API doit etre joignable via API_URL (defaut http://api:8000
en docker).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# Nombre de previsions envoyees a chaque execution.
N_PREDICTIONS = 20

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_send_predictions(**context) -> None:
    """Echantillonner N_PREDICTIONS lignes et les envoyer a l'API /predict."""
    import httpx

    from src.config import API_URL, TARGET
    from src.data import load_data

    # On retire la colonne cible : l'API ne recoit que les features.
    features = load_data().drop(columns=[TARGET])

    # Echantillonner N_PREDICTIONS lignes
    sample = features.sample(n=N_PREDICTIONS)

    # Envoyer les predictions a l'API
    with httpx.Client(base_url=API_URL, timeout=10.0) as client:
        client.get("/health").raise_for_status()
        for _, row in sample.iterrows():
            payload = json.loads(row.to_json())
            response = client.post("/predict", json=payload)
            response.raise_for_status()

    logger.info("%d previsions envoyees a %s", N_PREDICTIONS, API_URL)


with DAG(
    dag_id="daily_predictions",
    description="Envoie 20 previsions par jour a l'API (trafic simule)",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="0 10 * * *",  # Tous les jours a 10h
    catchup=False,
    tags=["classification", "predictions"],
) as dag:
    send_predictions = PythonOperator(
        task_id="send_predictions",
        python_callable=task_send_predictions,
    )
