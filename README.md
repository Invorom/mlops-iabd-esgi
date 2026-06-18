# 💳 SentinelAI — Détection de Fraude Bancaire (MLOps)

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-blueviolet.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API%20REST-green.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerization-blue.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Airflow](https://img.shields.io/badge/Apache%20Airflow-Orchestration-017CE2.svg?logo=apache-airflow&logoColor=white)](https://airflow.apache.org/)

Ce projet est une plateforme complète et conteneurisée de Machine Learning (MLOps) conçue pour détecter les fraudes aux cartes bancaires en temps réel. Développé par **Romain NEROT** dans le cadre du cursus ESGI.

---

## 📌 Architecture & Fonctionnalités

La plateforme intègre les meilleures pratiques MLOps avec un cycle de vie du modèle entièrement automatisé :
- **Entraînement & Tracking** : Utilisation de `MLflow` pour enregistrer les hyperparamètres, métriques (F1-Score, ROC-AUC) et conserver les modèles (`skops`/`cloudpickle`).
- **Orchestration Automatisée** : Des pipelines (DAGs) `Apache Airflow` assurent le ré-entraînement régulier du modèle et les prédictions par lots.
- **Déploiement API** : Une API robuste développée en `FastAPI` expose le meilleur modèle pour l'inférence en temps réel.
- **Interface Utilisateur (UI)** : Un dashboard interactif `Streamlit` avec un design "Glassmorphism" moderne pour simuler des transactions.
- **Conteneurisation Complète** : Tout l'écosystème tourne sous `Docker Compose` avec des images allégées (`uv` python-slim).

---

## 📊 Jeu de Données

Le projet utilise **100 000 transactions** provenant de [Kaggle](https://www.kaggle.com/datasets/anurag629/credit-card-fraud-transaction-data). Les features incluent :
- **Financier** : Montant, Type de transaction (POS, Online, ATM).
- **Géographie** : Pays de résidence, Pays de transaction.
- **Profil** : Âge, Genre.
- **Technique** : Banque émettrice, Réseau (Visa/MasterCard), Mode de saisie (PIN, Tap, CVC).

---

## 🛠️ Stack Technique Principale

*   **Gestionnaire de packages** : `uv` (ultra-rapide)
*   **Machine Learning** : `scikit-learn`, `XGBoost`, `LightGBM`
*   **Optimisation d'hyperparamètres** : `Optuna`
*   **Orchestration** : `Apache Airflow 3.x` (Standalone mode)
*   **Tracking** : `MLflow` (Backend SQLite)
*   **API & Frontend** : `FastAPI` (Uvicorn), `Streamlit`
*   **Déploiement** : `Docker Compose`

---

## 🚀 Mise en Route (Docker)

L'intégralité du projet peut être lancée avec Docker. Assurez-vous d'avoir téléchargé le fichier de données `dataset.csv` dans le dossier `data/`.

### 1. Démarrer les services principaux (API, Frontend, MLflow)
```bash
make docker-up
```
- 🌐 Frontend Streamlit : [http://localhost:8501](http://localhost:8501)
- ⚙️ API FastAPI : [http://localhost:8000/docs](http://localhost:8000/docs)
- 🧪 MLflow UI : [http://localhost:5000](http://localhost:5000)

### 2. Démarrer Apache Airflow
```bash
make airflow
```
- 🕒 Interface Airflow : [http://localhost:8080](http://localhost:8080)
- Identifiants : **Login : admin** / Le mot de passe est généré automatiquement. Pour le récupérer, tapez :
  ```bash
  make airflow-password
  ```

### 3. Workflow Local (Développement)
Si vous souhaitez développer localement sans Docker :
```bash
make install          # Installe les dépendances via uv
make train            # Entraîne le modèle baseline localement
make train-optuna     # Lance l'optimisation bayésienne
make check            # Lance les tests, le linter et le typage
```

---

## 📈 Suivi des Performances

Le système compare de multiples algorithmes (`Random Forest`, `XGBoost`, `LightGBM`) et sélectionne le meilleur basé sur une double validation croisée :
- Minimisation des **faux positifs** (garantir une bonne expérience client)
- Maximisation du **F1-Score** et du **ROC AUC** (capturer le maximum de fraudes)

L'évaluation et la validation du modèle sont gérées automatiquement par les pipelines Airflow avant toute mise en production.
