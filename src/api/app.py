from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import joblib
import numpy as np
import logging
import json

logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

model = joblib.load('models/KNN_250.joblib')
encoder = joblib.load('models/encoder.joblib')
app = FastAPI()

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

class LoginData(BaseModel):
    username: str
    password: str

def load_model_performance():
    with open('models/KNN250_performance.json', 'r') as f:
        return json.load(f)
    
def load_user_credentials():
    with open('data/users.json', 'r') as f:
        return json.load(f)

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API du modèle de Machine Learning de prédiction de la gravité des accidents de la route"}

@app.post("/predict")
def predict(data: AccidentData):
    try:
        # Extraire les données sous forme de liste avec les 16 variables
        input_data = np.array([[
            data.catu, data.catv, data.obsm, data.col, data.place,
            data.manv, data.situ, data.agg, data.plan, data.secu_combined,
            data.age_category_encoded, data.infra, data.inter, data.sexe,
            data.catr, data.lum
        ]])

        # Encoder les données
        input_data_encoded = encoder.transform(input_data).toarray()

        # Faire une prédiction avec le modèle
        prediction = model.predict(input_data_encoded)
    
        # Retourner la prédiction sous forme d'entier
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

@app.get("/model_performance")
def get_model_performance():
    model_performance_data = load_model_performance()
    return model_performance_data

# A voir lorsque "predict" fonctionnera

# @app.post("/login")
# def login(data: LoginData):
#     logging.info("Login request received for user: %s", data.username)
#     credentials = load_user_credentials()
    
#     if data.username in credentials and credentials[data.username] == data.password:
#         logging.info("User %s authenticated successfully", data.username)
#         return {"message": "Login successful"}
#     else:
#         logging.warning("Failed login attempt for user: %s", data.username)
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
# @app.get("/logs")
# def get_logs():
#     logging.info("Logs endpoint accessed")
#     try:
#         with open('app.log', 'r') as f:
#             logs = f.read()
#         return {"logs": logs}
#     except Exception as e:
#         logging.error("Failed to read logs: %s", str(e))
#         return {"error": "Failed to read logs"}