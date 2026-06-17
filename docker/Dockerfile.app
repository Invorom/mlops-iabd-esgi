FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Installation des dépendances (mise en cache)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copie de l'ensemble du projet
COPY . .

# Variables d'environnement par défaut
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV API_URL=http://api:8000

# Exposition des ports (indicatif, géré par le docker-compose)
EXPOSE 8000
EXPOSE 8501
EXPOSE 5000
