from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import uvicorn
import os
import joblib
import uvicorn
from fastapi.responses import Response
from prometheus_client import start_http_server, Gauge, generate_latest, CONTENT_TYPE_LATEST
import json
from evidently.report import Report
from evidently.metrics import DataDriftTable, DatasetDriftMetric
from evidently import ColumnMapping

'''
# Create monitoring folder
if not os.path.exists("monitoring"):
    os.makedirs("monitoring",exist_ok=True)

LIVE_DATA_PATH = "monitoring/live_data.csv"
REFERENCE_DATA_PATH = "monitoring/reference_data.csv"
'''

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
    #"logistic_regression": logistic_regression_model,
    #"random_forest": random_forest_model,
    "xgboost": xgboost_model
}

# Mount the static/css folder
app.mount("/static", StaticFiles(directory="static"), name="static")

'''
# Load reference data (keeping only features)
if not os.path.exists(REFERENCE_DATA_PATH):
    df = pd.read_csv("data/processed/creditcard_processed.csv")
    reference_features = df.drop(columns=["Class"])  # Drop target column
    reference_features.to_csv(REFERENCE_DATA_PATH, index=False)

# Column Mapping (for Evidently)
column_mapping = ColumnMapping()
column_mapping.target = None  # No target for data drift
column_mapping.prediction = "prediction"  # Only needed for prediction drift

# Evidently Reports
data_drift_report = Report(metrics=[DataDriftTable()])
prediction_drift_report = Report(metrics=[DatasetDriftMetric()])

# Prometheus Metrics
data_drift_detected = Gauge("data_drift_detected", "Data drift detected (1 = Yes, 0 = No)")
prediction_drift_detected = Gauge("prediction_drift_detected", "Prediction drift detected (1 = Yes, 0 = No)")
'''

@app.get("/")
async def read_root(request: Request):     
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request,features: str = Form(...), amount: str = Form(...), time: str = Form(...),model: str = Form(...)):


    amount = float(amount)
    time = float(time)

    amount_per_time = amount/ (time+1)

    # Convert the features into a list
    features = [float(x) for x in features.split(',')]

    scaled_amount = standard_scaler.transform([[amount]])[0][0]
    scaled_amount_per_time = min_max_scaler.transform([[amount_per_time]])[0][0]

    #Append the scaled amount and amount_per_time to features
    features.append(scaled_amount)
    features.append(scaled_amount_per_time)

    # Load the chosen model
    selected_model = models.get(model)

    # Predict the output
    #features_array = np.array(features).reshape(1,-1)
    #prediction = selected_model.predict(features_array)
    prediction = selected_model.predict([features])[0]
    threshold = 0.001
    #prediction_text = 'Fraud' if prediction[0] == 1 else 'Not Fraud'
    
    proba = selected_model.predict_proba([features])[0]

    prediction = 1 if proba[1] > threshold else 0

    prediction_text = 'Fraud' if prediction == 1 else 'Not Fraud'

    print(f"Prediction Probability: {proba}")

    '''
    # Store prediction data
    record = features + [prediction]
    save_to_csv(record)'
    '''

    return templates.TemplateResponse("index.html", {"request": request, "prediction_text": prediction_text})

'''
def save_to_csv(record):
    """Appends new prediction records to live_data.csv for monitoring."""
    feature_columns = [f"V{i+1}" for i in range(28)] + ["scaled_amount", "scaled_amount_per_time"]
    columns = feature_columns + ["prediction"]
    df = pd.DataFrame([record], columns=columns)

    if not os.path.exists(LIVE_DATA_PATH):
        df.to_csv(LIVE_DATA_PATH, index=False)
    else:
        df.to_csv(LIVE_DATA_PATH, mode="a", header=False, index=False)

@app.get("/data_drift")
async def detect_data_drift():
    """Runs Evidently Data Drift Report on feature columns only."""
    if not os.path.exists(LIVE_DATA_PATH):
        return {"error": "No data available for drift detection."}

    live_data = pd.read_csv(LIVE_DATA_PATH)
    reference_data = pd.read_csv(REFERENCE_DATA_PATH)

    # Only check feature columns for data drift
    feature_columns = [f"V{i+1}" for i in range(28)] + ["scaled_amount", "scaled_amount_per_time"]
    live_features = live_data[feature_columns]
    reference_features = reference_data[feature_columns]

    if len(live_features) < 10:
        return {"error": "Not enough data for drift detection."}

    data_drift_report.run(reference_data=reference_features, current_data=live_features, column_mapping=column_mapping)
    drift_result = data_drift_report.as_dict()["metrics"][0]["result"]["dataset_drift"]

    # Store report
    with open("monitoring/data_drift.json", "w") as f:
        json.dump(data_drift_report.as_dict(), f, indent=4)

    data_drift_detected.set(1 if drift_result else 0)  # Log to Prometheus
    return {"data_drift_detected": drift_result}


@app.get("/prediction_drift")
async def detect_prediction_drift():
    """Runs Evidently Prediction Drift Report on the `prediction` column."""
    if not os.path.exists(LIVE_DATA_PATH):
        return {"error": "No data available for prediction drift detection."}

    live_data = pd.read_csv(LIVE_DATA_PATH)
    reference_data = pd.read_csv(REFERENCE_DATA_PATH)

    if len(live_data) < 10:
        return {"error": "Not enough data for prediction drift detection."}

    # Only check the "prediction" column for prediction drift
    live_predictions = live_data[["prediction"]]
    reference_predictions = reference_data[["prediction"]]

    prediction_drift_report.run(
        reference_data=reference_predictions, current_data=live_predictions, column_mapping=column_mapping
    )
    drift_result = prediction_drift_report.as_dict()["metrics"][0]["result"]["dataset_drift"]

    # Store report
    with open("monitoring/prediction_drift.json", "w") as f:
        json.dump(prediction_drift_report.as_dict(), f, indent=4)

    prediction_drift_detected.set(1 if drift_result else 0)  # Log to Prometheus
    return {"prediction_drift_detected": drift_result}


@app.get("/metrics")
async def prometheus_metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Start Prometheus server
start_http_server(8001)  # Prometheus scrapes this port
'''


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
    