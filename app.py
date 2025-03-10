from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import os
import joblib
import pickle
import uvicorn

# Setup monitoring folder
if not os.path.exists("monitoring"):
    os.makedirs("monitoring", exist_ok=True)

if not os.path.exists("templates"):
    os.makedirs("templates", exist_ok=True)  

LIVE_DATA_PATH = "monitoring/live_data.csv"
REFERENCE_DATA_PATH = "monitoring/reference_data.csv"

app = FastAPI()
templates = Jinja2Templates(directory="templates") 

xgboost_path = "data/models/fraud_model.pkl"

# Load the trained scaler
min_max_scaler = joblib.load('data/models/min_max_scaler.pkl')
standard_scaler = joblib.load('data/models/standard_scaler.pkl')

# Load the trained model
xgboost_model = pickle.load(open(xgboost_path, "rb"))

# A dictionary to hold model selection
models = {
    "xgboost": xgboost_model
}

# Mount the static/css folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load reference data 
if not os.path.exists(REFERENCE_DATA_PATH):
    df = pd.read_csv("data/processed/creditcard_processed.csv")
    reference_features = df.drop(columns=["Class"]) 
    reference_features.to_csv(REFERENCE_DATA_PATH, index=False)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
#async def predict(request: Request, features: str = Form(...), amount: str = Form(...), time: str = Form(...), model: str = Form(...)):
async def predict(request: Request):
    '''
    amount = float(amount)
    time = float(time)
    amount_per_time = amount / (time + 1)'
    '''
    form_data = await request.form()
    input_data = form_data['features']  
    time = float(form_data['time'])
    amount = float(form_data['amount'])
    model = form_data['model']

    # Convert the features into a list
    features = [float(x) for x in input_data.split(',')]

    amount_per_time = amount / (time + 1)

    # Scale the values
    scaled_amount = standard_scaler.transform(np.array([[amount]]))[0][0]
    scaled_amount_per_time = min_max_scaler.transform(np.array([[amount_per_time]]))[0][0]

    # Append the scaled amount and amount_per_time to features
    features.append(scaled_amount)
    features.append(scaled_amount_per_time)

    # Load the chosen model
    selected_model = models.get(model)

    # Predict the output
    prediction = selected_model.predict([features])[0]
    threshold = 0.001
    proba = selected_model.predict_proba([features])[0]
    prediction = 1 if proba[1] > threshold else 0
    prediction_text = 'Fraud' if prediction == 1 else 'Not Fraud'

    # Store prediction data
    save_to_csv(features)

    return templates.TemplateResponse("index.html", {"request": request, "prediction_text": prediction_text})


def save_to_csv(record):
    """Appends new prediction records to live_data.csv for monitoring."""
    feature_columns = [f"V{i+1}" for i in range(28)] + ["scaled_amount", "scaled_amount_per_time"]
    df = pd.DataFrame([record], columns=feature_columns)

    if not os.path.exists(LIVE_DATA_PATH):
        df.to_csv(LIVE_DATA_PATH, index=False)
    else:
        df.to_csv(LIVE_DATA_PATH, mode="a", header=False, index=False)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
