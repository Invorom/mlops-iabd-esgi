"""Configuration centrale du projet de classification.

C'est le SEUL fichier a adapter pour brancher votre propre jeu de donnees :
data.py, features.py et les scripts d'entrainement lisent toutes leurs
colonnes via ces constantes. Voir tp/TP_S0_projet_personnel.md.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# (S0-1) : chemin vers votre fichier de donnees (CSV) place dans data/
DATA_PATH = ROOT / "data" / "dataset.csv"
MODEL_DIR = ROOT / "models"

# (S0-2) : nom de la colonne cible binaire (valeurs 0/1)
TARGET = "Fraud"

# (S0-3) : colonnes numeriques de votre dataset
NUMERIC_FEATURES: list[str] = [
    "Amount",
    "Age",
]

# (S0-4) : colonnes categorielles (les autres colonnes non listees seront ignorees/dropped)
CATEGORICAL_FEATURES: list[str] = [
    "Day of Week",
    "Type of Card",
    "Entry Mode",
    "Type of Transaction",
    "Merchant Group",
    "Country of Transaction",
    "Country of Residence",
    "Gender",
    "Bank",
]

RANDOM_STATE = 42

# Surcouche via variables d'environnement (principe 12-factor)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "credit-card-fraud")
MLFLOW_EXPERIMENT_DESCRIPTION = "Détection de fraudes par carte bancaire"
MLFLOW_EXPERIMENT_TAGS = {"project_type": "classification", "dataset": "kaggle_fraud"}
MODEL_NAME = os.getenv("MODEL_NAME", "fraud-detector")
