from fastapi import FastAPI
import pickle
import numpy as np
import uvicorn
import os

app = FastAPI()

# Load the trained model
path = "data/models/fraud_model.pkl"

model = pickle.load(open(path, "rb"))

@app.post("/predict")
def predict(data:dict):

    features = np.array(data["features"]).reshape(1,-1)
    prediction = model.predict(features)
    return {"fraud_prediction": int(prediction[0])}
    