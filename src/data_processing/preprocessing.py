import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

def process(input_path,output_path,scaler_path):
    '''Processes the raw data'''
    
    df = pd.read_csv(input_path)

    # Drop the null values if any
    df.dropna(inplace=True)

    # Fix the outliers
    df['Amount'] = np.log1p(df['Amount'])

    # Normalize Amount
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])

    df.to_csv(output_path,index=False)

    # Save the scaler to a .pkl file
    joblib.dump(scaler,scaler_path)
    print(f"Scaler saved to {scaler_path}")

    print(f" Preprocessing completed and data saved to {output_path}")


if __name__=='__main__':

    input_path = "../../data/raw/creditcard.csv"
    output_path = "../../data/processed/creditcard_processed.csv"
    scaler_path = "../../data/models/standard_scaler.pkl"

    process(input_path,output_path,scaler_path)

