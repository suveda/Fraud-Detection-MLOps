import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import  StandardScaler,MinMaxScaler

def feature_eng(input_path,output_path,scaler_path):
    ''' processes feature engineering'''

    df = pd.read_csv(input_path)

    # Create amount/time column
    df['Amount_per_time'] = df['Amount'] / (df['Time'] + 1)

    # Drop the time column
    df.drop(columns=['Time'], inplace=True)

    # Normalize the new column
    #scaler = StandardScaler()
    scaler = MinMaxScaler()
    df['Amount_per_time'] = scaler.fit_transform(df[['Amount_per_time']])

    df.to_csv(output_path,index=False)

    # Save the scaler to a .pkl file
    joblib.dump(scaler,scaler_path)
    print(f"Scaler saved to {scaler_path}")

    print(f" Feature engineering completed and saved to {output_path}")

if __name__=='__main__':

    input_path = "../../data/processed/creditcard_processed.csv"
    output_path = "../../data/processed/creditcard_featured.csv"
    scaler_path = "../../data/models/min_max_scaler.pkl"

    feature_eng(input_path,output_path,scaler_path)