# 💳 Détection de Fraude Bancaire — Projet MLOps (IABD ESGI)

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-blueviolet.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API%20REST-green.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerization-blue.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Airflow](https://img.shields.io/badge/Apache%20Airflow-Orchestration-017CE2.svg?logo=apache-airflow&logoColor=white)](https://airflow.apache.org/)

Ce dépôt héberge le projet fil rouge de détection de fraude sur cartes bancaires développé dans le cadre du module d'orchestration et de MLOps à l'ESGI. Il s'appuie sur le squelette générique de classification binaire du dossier [`todo/`](file:///c:/Users/romai/OneDrive/Documents/GitHub/mlops-iabd-esgi/todo).

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

Contrairement aux datasets de fraude classiques basés sur des variables anonymisées par PCA, ce jeu de données contient des attributs réels et exploitables pour la modélisation :

| Catégorie | Variables | Description |
|---|---|---|
| **Opération** | `Transaction ID`, `Date`, `Time`, `Day of Week` | Identifiants et métadonnées temporelles |
| **Transaction** | `Amount`, `Type of Transaction`, `Entry Mode` | Montant, canal (Online, POS, ATM) et mode d'authentification (Tap, PIN, CVC) |
| **Profil Client** | `Gender of Cardholder`, `Age of Cardholder` | Données démographiques du titulaire |
| **Cartographie**| `Type of Card`, `Issuing Bank` | Réseau (Visa/MasterCard) et banque émettrice |
| **Localisation** | `Transaction Country`, `Shipping Address`, `Billing Address` | Pays de la transaction et adresses de livraison/facturation |
| **Cible (Target)** | `is_fraud` (ou variable cible binaire) | Indique si la transaction est frauduleuse (`1`) ou légitime (`0`) |

---

## 🛠️ Stack Technique

*   **Gestionnaire d'environnement** : `uv` (Python 3.13)
*   **Suivi & Registre de modèles** : `MLflow` (via Docker Compose)
*   **Interface Applicative (API)** : `FastAPI` (Uvicorn)
*   **Interface Utilisateur (Frontend)** : `Streamlit`
*   **Orchestration & Workflow** : `Apache Airflow` (DAGs de réentraînement)
*   **Conteneurisation** : `Docker` & `Docker Compose`
*   **CI/CD** : `GitHub Actions`

---

## 📂 Arborescence du Projet

```
mlops-iabd-esgi/
  README.md                 Ce fichier de présentation du projet
  .gitignore                Exclusions Git (secrets, caches, artefacts de modèles)
  slides/                   Supports de cours MLOps et d'orchestration
  tp/                       Énoncés détaillés des séances de TP
  data/                     Dossier destiné à accueillir le fichier CSV de Kaggle
  todo/                     Code source et squelette du projet à compléter
    Makefile                Commandes d'automatisation (installation, tests, run...)
    mlproject/              Package Python principal contenant les modules de ML
      config.py             Configuration centrale (features, target, chemins)
      data.py               Chargement et découpage train/test stratifié
      features.py           Pipeline de pré-traitement des données (ColumnTransformer)
      evaluation.py         Génération de graphes d'évaluation (dont SHAP)
      train.py              Entraînement de la baseline et tracking MLflow
      train_optuna.py       Optimisation d'hyperparamètres et enregistrement MLflow
      train_models.py       Comparaison de modèles avec GridSearchCV et SHAP
      api.py                API de prédiction FastAPI
    frontend/               Frontend interactif Streamlit
    docker/                 Fichiers Docker pour l'entraînement, l'API et le frontend
    docker-compose.yml      Orchestration locale des services
    dags/                   DAGs Airflow pour le réentraînement programmé
```

---

## 🚀 Mise en Route rapide

### 1. Installation de l'environnement
Depuis la racine du projet, installez les dépendances avec `uv` via le Makefile :
```bash
make -C todo install
```

### 2. Configuration du Dataset
1. Téléchargez le fichier CSV sur [Kaggle](https://www.kaggle.com/datasets/anurag629/credit-card-fraud-transaction-data).
2. Placez le fichier dans le dossier `data/` et renommez-le en `dataset.csv` (ou ajustez le nom dans la configuration).
3. Modifiez les constantes dans [`todo/mlproject/config.py`](file:///c:/Users/romai/OneDrive/Documents/GitHub/mlops-iabd-esgi/todo/mlproject/config.py) pour y renseigner le nom de la colonne cible ainsi que vos listes de caractéristiques numériques et catégorielles.

### 3. Exécution locale
Définissez le chemin Python et lancez l'entraînement baseline :
```bash
# Windows (PowerShell)
$env:PYTHONPATH="todo"
python -m mlproject.train

# Linux/macOS
PYTHONPATH=todo python -m mlproject.train
```

---

## 📅 Feuille de Route & Progression

Chaque étape du projet correspond à un TP et est matérialisée par des marqueurs `TODO (Sx-n)` dans le code du dossier `todo/` :

| Séance | TP Associé | Fichier cible | Objectif MLOps |
| :---: | :--- | :--- | :--- |
| **S0** | `tp/TP_S0_projet_personnel.md` | [`todo/mlproject/config.py`](file:///c:/Users/romai/OneDrive/Documents/GitHub/mlops-iabd-esgi/todo/mlproject/config.py) | Initialiser et configurer le dataset de fraude |
| **S5** | `tp/TP_S5_mlflow.md` | [`todo/mlproject/train.py`](file:///c:/Users/romai/OneDrive/Documents/GitHub/mlops-iabd-esgi/todo/mlproject/train.py) | Mettre en œuvre le tracking d'expériences MLflow |
| **S6** | `tp/TP_S6_optuna.md` | [`todo/mlproject/train_optuna.py`](file:///c:/Users/romai/OneDrive/Documents/GitHub/mlops-iabd-esgi/todo/mlproject/train_optuna.py) | Optimiser avec Optuna et enregistrer le modèle |
| **S7** | `tp/TP_S7_automl_shap.md` | [`todo/mlproject/train_models.py`](file:///c:/Users/romai/OneDrive/Documents/GitHub/mlops-iabd-esgi/todo/mlproject/train_models.py) | Comparer les modèles et analyser l'explicabilité (SHAP) |
| **S8** | `tp/TP_S8_docker.md` | `todo/docker/Dockerfile.train` | Conteneuriser le script d'entraînement |
| **S12** | `tp/TP_S12_fastapi.md` | [`todo/mlproject/api.py`](file:///c:/Users/romai/OneDrive/Documents/GitHub/mlops-iabd-esgi/todo/mlproject/api.py) | Exposer le meilleur modèle via une API FastAPI |
| **S14** | `tp/TP_S14_docker_compose.md` | `todo/docker-compose.yml` | Orchestrer les conteneurs (API, MLflow, Frontend) |
| **S14b**| `tp/TP_S14_bis_streamlit.md` | `todo/frontend/app.py` | Développer l'interface client de simulation Streamlit |
| **S17** | `tp/TP_S17_airflow.md` | `todo/dags/retrain_dag.py` | Planifier le pipeline de réentraînement avec Airflow |

---

## 📈 Suivi & Collaboration

*   Les commits doivent être réguliers à chaque séance pour assurer un bon suivi de la progression.
*   Ajoutez l'enseignant en tant que collaborateur sur votre dépôt GitHub : [lewishkpv](https://github.com/lewishkpv).
*   Ne poussez jamais de données volumineuses (`.csv`), d'environnements virtuels (`.venv`) ou d'artefacts MLflow (`mlruns/`, `*.joblib`) grâce aux règles déjà présentes dans le `.gitignore`.
