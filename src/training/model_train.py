import pandas as pd
import xgboost as xgb
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

def train(input_path,model_path,hyperparam_path):
    '''Trains the model'''

    df = pd.read_csv(input_path)

    # Load the best hyper-parameters
    best_params = pickle.load(open(hyperparam_path,"rb"))
    
    # Create feature and target
    X = df.drop(columns=['Class'],axis=1)
    y = df['Class']

    # Split the data into training and test sets
    X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    # Initalize XGB classifier and train
    model = xgb.XGBClassifier(**best_params, random_state=42)
    #model = RandomForestClassifier(random_state=0)
    model.fit(X_train,y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Display the clssification report and confusion matrix
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Classification Report: {cr}")
    print(f"Confusion Matrix: {cm}")

    # save the trained model
    pickle.dump(model, open(model_path, "wb"))
    print(f"Saved the model to {model_path}")

if __name__=='__main__':

    input_path = "../../data/processed/creditcard_processed.csv"
    model_path = "../../data/models/fraud_model.pkl"
    hyperparam_path = "../../data/models/best_hyperparams.pkl"

    train(input_path,model_path,hyperparam_path)


