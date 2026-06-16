"""API d'inference du modele de detection de fraude (FastAPI).

Seance 12 - TP FastAPI
    Lancement : uvicorn src.api:app --reload
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Charger le modele au demarrage, le liberer a l'arret."""
    model_path = MODEL_DIR / "model.joblib"
    logger.info("Chargement du modele depuis %s", model_path)
    ml["model"] = joblib.load(model_path)
    logger.info("Modele charge avec succes.")
    yield
    ml.clear()
    logger.info("Modele decharge.")


app = FastAPI(title="Fraud Detection API", version="0.1.0", lifespan=lifespan)


class Features(BaseModel):
    """Schema d'entree correspondant aux features du dataset Credit Card Fraud."""

    Amount: float = Field(..., description="Montant de la transaction")
    Age: float = Field(..., ge=0, description="Age du titulaire de la carte")
    Day_of_Week: str = Field(..., alias="Day of Week", description="Jour de la semaine")
    Type_of_Card: str = Field(..., alias="Type of Card", description="Type de carte")
    Entry_Mode: str = Field(..., alias="Entry Mode", description="Mode d'entree")
    Type_of_Transaction: str = Field(..., alias="Type of Transaction", description="Type de transaction")
    Merchant_Group: str = Field(..., alias="Merchant Group", description="Groupe du marchand")
    Country_of_Transaction: str = Field(..., alias="Country of Transaction", description="Pays de la transaction")
    Country_of_Residence: str = Field(..., alias="Country of Residence", description="Pays de residence")
    Gender: str = Field(..., description="Genre du titulaire")
    Bank: str = Field(..., description="Banque emettrice")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "Amount": 150.0,
                    "Age": 35,
                    "Day of Week": "Monday",
                    "Type of Card": "Visa",
                    "Entry Mode": "Tap",
                    "Type of Transaction": "POS",
                    "Merchant Group": "Electronics",
                    "Country of Transaction": "United Kingdom",
                    "Country of Residence": "United Kingdom",
                    "Gender": "M",
                    "Bank": "HSBC",
                }
            ]
        },
    }


class PredictionOut(BaseModel):
    """Schema de sortie de la prediction."""

    prediction: int = Field(..., description="Classe predite (0=legitime, 1=fraude)")
    probability: float = Field(..., description="Probabilite de fraude")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    row = pd.DataFrame([features.model_dump(by_alias=True)])
    proba = float(model.predict_proba(row)[0, 1])
    return PredictionOut(prediction=int(proba >= 0.5), probability=round(proba, 4))


@app.get("/model-info")
def model_info() -> dict:
    return {"version": os.environ.get("MODEL_VERSION", "unknown")}
