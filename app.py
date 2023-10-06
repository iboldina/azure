import uvicorn
from fastapi import FastAPI
from joblib import load
import sklearn
import lightgbm

#from pydantic import  BaseModel

import pandas as pd
import numpy as np

app = FastAPI(title="Modèle de scoring", description="API pour le modèle de scoring")
# Load the data
data = pd.read_csv('test_production.csv')
# Define a global variable for the model
model = None
@app.on_event('startup')
def load_model():
    # Use the global keyword to indicate that we are using the global model variable
    global model
    # Load the model
    model = load('model.joblib')

@app.post("/predict/{id}")
def predict(id: int):
    # Select the row where SK_ID_CURR equals the provided id
    row = data[data['SK_ID_CURR'] == id]

    # Use the model to make a prediction on this row
    prediction = model.predict(row)
    log_proba = model.predict_proba(row)


    return {"prediction": prediction.tolist(),
            "log_proba": log_proba.tolist()}

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:5000


# uvicorn app:app --reload
# uvicorn app:app --port 5000

#predict(data)
#curl -X POST http://127.0.0.1:5000/predict (to run in terminal)
# curl -X POST http://127.0.0.1:5000/predict/420839
