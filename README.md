# 💳 Détection de Fraude Bancaire — Projet MLOps (IABD ESGI)

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-blueviolet.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API%20REST-green.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerization-blue.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Airflow](https://img.shields.io/badge/Apache%20Airflow-Orchestration-017CE2.svg?logo=apache-airflow&logoColor=white)](https://airflow.apache.org/)

Ce dépôt héberge le projet fil rouge de détection de fraude sur cartes bancaires développé dans le cadre du module d'orchestration et de MLOps à l'ESGI.

---

## 📌 Problématique & Contexte

Chaque année, les fraudes à la carte bancaire causent des pertes financières de plusieurs milliards. Si le coût direct de la fraude est colossal pour le système financier, **le coût lié à sa détection l'est tout autant**. 

Dans un marché bancaire ultra-concurrentiel, les banques doivent arbitrer un dilemme majeur :
*   **Le risque de fraude** (Pertes financières, coûts d'assurance, démarches de remboursement).
*   **L'expérience client** (Impact des contrôles intrusifs et des faux positifs).

Un taux élevé de **faux positifs** signifie que des transactions légitimes de clients fidèles sont déclinées par erreur. Cette friction peut pousser le client à utiliser une carte concurrente ou à quitter sa banque.

> [!IMPORTANT]
> **Le défi MLOps** : Concevoir et déployer un système de détection robuste, performant et automatisé permettant d'atteindre un taux d'erreur minimal (en particulier sur les faux positifs) tout en assurant un réentraînement régulier et une haute disponibilité de l'API de prédiction.

---

## 📊 Jeu de Données

Le projet utilise le jeu de données Kaggle suivant :
🔗 [Credit Card Fraud Transaction Data - Anurag Verma](https://www.kaggle.com/datasets/anurag629/credit-card-fraud-transaction-data)

**100 000 transactions** avec des attributs réels et exploitables :

| Catégorie | Variables | Description |
|---|---|---|
| **Opération** | `Transaction ID`, `Date`, `Time`, `Day of Week` | Identifiants et métadonnées temporelles |
| **Transaction** | `Amount`, `Type of Transaction`, `Entry Mode` | Montant, canal (Online, POS, ATM) et mode d'authentification (Tap, PIN, CVC) |
| **Profil Client** | `Gender`, `Age` | Données démographiques du titulaire |
| **Cartographie** | `Type of Card`, `Bank` | Réseau (Visa/MasterCard) et banque émettrice |
| **Localisation** | `Country of Transaction`, `Country of Residence` | Pays de la transaction et pays de résidence |
| **Cible** | `Fraud` | `1` = frauduleuse, `0` = légitime |

---

## 🛠️ Stack Technique

*   **Gestionnaire d'environnement** : `uv` (Python 3.13)
*   **Suivi & Registre de modèles** : `MLflow` (tracking, model registry)
*   **Optimisation** : `Optuna` (TPE Sampler) + `GridSearchCV`
*   **Modèles** : `Logistic Regression`, `Decision Tree`, `Random Forest`, `XGBoost`, `LightGBM`
*   **Interface Applicative (API)** : `FastAPI` (Uvicorn)
*   **Conteneurisation** : `Docker` & `Docker Compose`
*   **CI/CD** : `GitHub Actions`

---

## 📂 Arborescence du Projet

```
mlops-iabd-esgi/
  README.md                 Ce fichier de présentation du projet
  Makefile                  Commandes d'automatisation (install, train, api...)
  pyproject.toml            Dépendances et métadonnées du projet
  .env                      Variables d'environnement (MLflow URI, etc.)
  .gitignore                Exclusions Git (secrets, caches, artefacts)
  data/
    dataset.csv             Jeu de données Kaggle (non versionné)
  models/
    model.joblib            Meilleur modèle persisté (non versionné)
  src/                      Package Python principal
    config.py               Configuration centrale (features, target, chemins, MLflow)
    data.py                 Chargement, nettoyage et découpage train/test stratifié
    features.py             Pipeline de pré-traitement (ColumnTransformer + Imputer)
    evaluation.py           Génération de graphes d'évaluation (SHAP summary plots)
    tracking.py             Configuration centralisée du tracking MLflow
    train.py                Entraînement baseline (Logistic Regression) + tracking MLflow
    train_models.py         Comparaison de modèles (GridSearchCV) : DT, RF, XGB, LGBM
    train_optuna.py         Optimisation bayésienne (Optuna TPE) : RF, XGB, LGBM
    evaluate.py             Évaluation automatisée et porte qualité (Quality Gate)
    api.py                  API de prédiction FastAPI (/health, /predict, /model-info)
    scripts/
      predict_fraud.py      Client de test pour l'API
```

---

## 🚀 Mise en Route rapide

### 1. Installation de l'environnement
```bash
make install
```

### 2. Configuration du Dataset
1. Téléchargez le fichier CSV sur [Kaggle](https://www.kaggle.com/datasets/anurag629/credit-card-fraud-transaction-data).
2. Placez le fichier dans le dossier `data/` et renommez-le en `dataset.csv`.

### 3. Lancer MLflow
Dans un terminal séparé :
```bash
uv run mlflow ui --port 5000
```
Interface accessible sur : http://127.0.0.1:5000

### 4. Entraînement

```bash
# Baseline (Logistic Regression)
make train

# Comparaison de modèles (GridSearchCV : DT, RF, XGB, LGBM)
uv run python -m src.train_models

# Optimisation bayésienne (Optuna TPE : RF, XGB, LGBM)
uv run python -m src.train_optuna --n-trials 15
```

### 5. Évaluation & Quality Gate
```bash
uv run python -m src.evaluate
```
Le modèle est validé contre des seuils minimaux configurables (`EVAL_F1_MIN=0.70`, `EVAL_ROC_AUC_MIN=0.90`).

### 6. API de prédiction
```bash
# Lancer l'API
uv run uvicorn src.api:app --reload

# Tester avec le client
uv run python -m src.scripts.predict_fraud
```
Documentation interactive : http://127.0.0.1:8000/docs

---

## 📊 Résultats

Les modèles sont comparés sur le jeu de test (20% stratifié) :

| Modèle | F1 Score | ROC AUC |
|---|---|---|
| Logistic Regression (baseline) | ~0.755 | ~0.962 |
| Decision Tree | optimisé via GridSearchCV | optimisé via GridSearchCV |
| Random Forest | optimisé via GridSearchCV / Optuna | optimisé via GridSearchCV / Optuna |
| XGBoost | optimisé via GridSearchCV / Optuna | optimisé via GridSearchCV / Optuna |
| LightGBM | optimisé via GridSearchCV / Optuna | optimisé via GridSearchCV / Optuna |

> [!TIP]
> Consultez l'interface MLflow (http://127.0.0.1:5000) pour visualiser les résultats détaillés, comparer les runs et explorer les artefacts (matrices de confusion, SHAP plots, rapports de classification).

---

## 📈 Suivi & Collaboration

*   Les commits doivent être réguliers à chaque séance pour assurer un bon suivi de la progression.
*   Ajoutez l'enseignant en tant que collaborateur sur votre dépôt GitHub : [lewishkpv](https://github.com/lewishkpv).
*   Ne poussez jamais de données volumineuses (`.csv`), d'environnements virtuels (`.venv`) ou d'artefacts MLflow (`mlruns/`, `*.joblib`) grâce aux règles déjà présentes dans le `.gitignore`.
