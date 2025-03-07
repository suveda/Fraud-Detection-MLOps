import pandas as pd
import xgboost as xgb
import os
import pickle
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def hyper_params(input_path,hyperparam_path):

    df = pd.read_csv(input_path)

    # Create feature and target
    X = df.drop(columns=['Class'],axis=1)
    y = df['Class']

    # Split the data into training and test sets
    X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

    def objective(trial):
        """Objective function for Optuna to find the best hyperparameters"""

        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        }

        model = xgb.XGBClassifier(**params,random_state=42)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        return accuracy_score(y_test,y_pred)
    
    # Run hyperparameter tuning 
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Get best parameters
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    # Save the best hyperparameters for future training
    pickle.dump(best_params,open(hyperparam_path,"wb"))
    print(f"Best hyperparameters saved at {hyperparam_path}")

if __name__=='__main__':

    input_path = "../../data/processed/creditcard_featured.csv"
    hyperparam_path = "../../data/models/best_hyperparams.pkl"

    hyper_params(input_path,hyperparam_path)