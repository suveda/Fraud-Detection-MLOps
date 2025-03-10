import pandas as pd
import xgboost as xgb
import os
import pickle
import numpy as np
import optuna
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from data_ingestion import load_and_preprocess
from imblearn.over_sampling import SMOTE

def hyperparameter_tuning(X_train, X_test , y_train, y_test):
    '''
    Tunes hyperparameters and returns the best parameters
    '''

    def tuning(trial):
        """Runs Optuna to find the best hyperparameters"""

        # Compute the scale_pos_weight for class imbalance
        negative_class = sum(y_train == 0)  
        positive_class = sum(y_train == 1)  
        scale_pos_weight = negative_class / positive_class

        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': trial.suggest_loguniform('scale_pos_weight', scale_pos_weight, scale_pos_weight*2)
        }

        model = xgb.XGBClassifier(**params,random_state=42)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        return accuracy_score(y_test,y_pred)
    
    # Run hyperparameter tuning 
    study = optuna.create_study(direction='maximize')
    study.optimize(tuning, n_trials=50)

    # Retreive the best parameters
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    return best_params

def save_best_params(best_params,param_path):
    '''
    Saves the best hyperparameters to a .pkl file
    '''
    
    pickle.dump(best_params,open(param_path,"wb"))
    print(f"Best hyperparameters saved at {param_path}")

def train_model(X_train,y_train,best_params):
    '''
    Trains the model
    '''
    
    # Initalize XGB classifier and train
    model = xgb.XGBClassifier(**best_params, random_state=42)
    model.fit(X_train,y_train)
    return model

def evaluate_model(model,X_test,y_test):
    '''
    Evaluates the models and returns the metric
    '''

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Classification Report: {cr}")
    print(f"Confusion Matrix: {cm}")

    return {"accuracy":accuracy,"classification_report":cr,"confusion_matrix":cm}

def save_model(model,model_path):
    '''
    Save the model to a .pkl file
    '''

    pickle.dump(model, open(model_path, "wb"))
    print(f"Saved the model to {model_path}")

def mlflow_logging(model,metrics,best_params):
    '''
    Logs the model, its parameters and metrics to mlflow
    '''
    experiment_name = "Fraud_Detection"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.sklearn.log_model(model, "fraud_detection_model")
        mlflow.end_run()

    print(f"Model logged in MLflow with accuracy: {metrics['accuracy']:.4f}")

def main():

    input_path = "../data/raw/creditcard.csv"

    param_path = "../data/models/best_hyperparams.pkl"

    model_path = "../data/models/fraud_model.pkl"

    # Load and preprocess the data
    data = load_and_preprocess(input_path)

    # Split the data into feature and target
    X = data.drop(columns=['Class'],axis=1)
    y = data['Class']

    X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

    # Aply smote to balance the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    best_params = hyperparameter_tuning(X_train_resampled, X_test , y_train_resampled, y_test)

    save_best_params(best_params,param_path)

    model = train_model(X_train_resampled,y_train_resampled,best_params)

    save_model(model,model_path)

    metrics = evaluate_model(model,X_test,y_test)

    mlflow_logging(model,metrics,best_params)


if __name__=="__main__":
    main()