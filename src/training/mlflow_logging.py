
import mlflow
import mlflow.sklearn
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

def flow(input_path,model_path):

    # Load the trained  model
    model = pickle.load(open(model_path,"rb"))

    # Load dataset to evaluate performance
    df = pd.read_csv(input_path)
    X = df.drop(columns=['Class'], axis=1)
    y = df['Class']

    # Get model predictions
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)

    # Start MLflow logging
    mlflow.start_run()
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "fraud_detection_model")
    mlflow.end_run()

    print(f"Model logged in MLflow with accuracy: {accuracy:.4f}")


if __name__=="__main__":

    input_path = "../../data/processed/creditcard_featured.csv"
    model_path = "../../data/models/fraud_model.pkl"

    flow(input_path,model_path)