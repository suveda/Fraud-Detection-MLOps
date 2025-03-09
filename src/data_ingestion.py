import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler,MinMaxScaler

def load_data(file_path):
    '''
    Loads the data from a CSV file
    '''

    data = pd.read_csv(file_path)
    print(f"Data loaded successfully from {file_path}")
    return data

def preprocess_data(data,standard_scaler_path="../data/models/standard_scaler.pkl",minmax_scaler_path="../data/models/min_max_scaler.pkl"):
    '''
    Preprocesses the data and performs feature engineering
    '''

    # Check and drop missing values
    data.dropna(inplace=True)

    # Fix the outliers
    data['Amount'] = np.log1p(data['Amount'])

    # Create a new amount/time column
    data['Amount_per_time'] = data['Amount'] / (data['Time'] + 1)

    # Drop the original time column
    data.drop(columns=['Time'], inplace=True)

    # Normalize Amount and Amount_per_time column
    standard_scaler = StandardScaler()
    min_max_scaler = MinMaxScaler()

    data['scaled_amount'] = standard_scaler.fit_transform(data[['Amount']])
    data['scaled_amount_per_time'] = min_max_scaler.fit_transform(data[['Amount_per_time']])

    # Drop the original 'Amount' and 'Amount_per_time' columns
    data = data.drop(['Amount', 'Amount_per_time'], axis=1)

    # Save the scaler to a .pkl file
    joblib.dump(standard_scaler,standard_scaler_path)
    joblib.dump(min_max_scaler,minmax_scaler_path)
    print(f"Scalers saved to {standard_scaler_path} and {minmax_scaler_path}")

    print("Data preprocessing and feature engineering completed")

    return data

def load_and_preprocess(file_path):
    '''
    Loads and preprocesses the data
    '''

    data = load_data(file_path)
    processed_data = preprocess_data(data)
    return processed_data


if __name__=='__main__':

    input_path = "../data/raw/creditcard.csv"

    processed_data = load_and_preprocess(input_path)

    # Save the processed data
    processed_data.to_csv("../data/processed/creditcard_processed.csv",index=False)


