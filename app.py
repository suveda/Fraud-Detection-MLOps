from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
import uvicorn
import os
import joblib
import uvicorn

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
    features_array = np.array(features).reshape(1,-1)
    prediction = selected_model.predict(features_array)
    prediction_text = 'Fraud' if prediction[0] == 1 else 'Not Fraud'

    return templates.TemplateResponse("index.html", {"request": request, "prediction_text": prediction_text})


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
    