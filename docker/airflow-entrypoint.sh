#!/bin/bash
set -e

# Initialiser la base Airflow si elle n'existe pas
if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
    echo ">> Initialisation de la base Airflow..."
    uv run airflow db migrate
    echo ">> Creation de l'utilisateur admin..."
    uv run airflow users create \
        --username admin \
        --password admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com
fi

# Lancer le scheduler en arrière-plan
echo ">> Demarrage du scheduler Airflow..."
uv run airflow scheduler &

# Lancer le webserver au premier plan
echo ">> Demarrage du webserver Airflow sur le port 8080..."
exec uv run airflow webserver --port 8080
