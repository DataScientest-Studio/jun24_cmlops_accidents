import json
import logging
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from src.api.auth import router as auth_router, authenticate
from src.api.log_module import router as log_router

from pydantic import BaseModel

# Tentative de chargement des modèles (gestion des erreurs si les fichiers manquent)
try:
    model = joblib.load("models/KNN_250.joblib")
    encoder = joblib.load("models/encoder.joblib")
    logging.info("Model and encoder loaded successfully.")
except Exception as e:
    model = None
    encoder = None
    logging.error(f"Error loading model or encoder: {e}")

app = FastAPI()

security = HTTPBasic()

class AccidentData(BaseModel):
    catu: int
    catv: int
    obsm: int
    col: int
    place: int
    manv: int
    situ: int
    agg: int
    plan: int
    secu_combined: int
    age_category_encoded: int
    infra: int
    inter: int
    sexe: int
    catr: int
    lum: int


def load_model_performance():
    try:
        with open("models/KNN250_performance.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error("Performance file not found.")
        raise HTTPException(status_code=500, detail="Performance data not available.")

# Route d'accueil
@app.get("/")
def home():
    return {
        "message": "Bienvenue sur l'API du modèle de prédiction de la gravité des accidents de la route"
    }


@app.post("/predict")
def predict(data: AccidentData, credentials: HTTPBasicCredentials = Depends(security)):
    # Authentification
    authenticate(credentials)

    if model is None or encoder is None:
        logging.error("Model or encoder not loaded.")
        raise HTTPException(status_code=500, detail="Model or encoder not loaded.")

    try:
        input_data = pd.DataFrame(
            [
                [
                    data.catu,
                    data.catv,
                    data.obsm,
                    data.col,
                    data.place,
                    data.manv,
                    data.situ,
                    data.agg,
                    data.plan,
                    data.secu_combined,
                    data.age_category_encoded,
                    data.infra,
                    data.inter,
                    data.sexe,
                    data.catr,
                    data.lum,
                ]
            ], columns=[
                'catu', 'catv', 'obsm', 'col', 'place', 'manv', 'situ', 'agg', 
            'plan', 'secu_combined', 'age_category_encoded', 'infra', 'inter', 
            'sexe', 'catr', 'lum'
            ]
        )

        logging.info(f"Received input data: {input_data}")
        logging.info(f"Data types of the input data: {input_data.dtypes}")
        logging.info(f"Input data shape before encoding: {input_data.shape}")

        input_data = input_data.astype('object')
        input_data_encoded = encoder.transform(input_data)
        logging.info(f"Encoded input data: {input_data_encoded}")

        prediction = model.predict(input_data_encoded)
        logging.info(f"Prediction result: {prediction}")

        return {"prediction": int(prediction[0])}

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")


@app.get("/model_performance")
def get_model_performance(credentials: HTTPBasicCredentials = Depends(security)):
    authenticate(credentials)

    model_performance_data = load_model_performance()
    return model_performance_data


# Inclusion des routes
app.include_router(auth_router, prefix="/auth")
app.include_router(log_router, prefix="/logs")
