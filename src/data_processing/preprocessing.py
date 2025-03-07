import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def process(input_path,output_path):
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

    print(f" Preprocessing completed and data saved to {output_path}")


if __name__=='__main__':

    input_path = "../../data/raw/creditcard.csv"
    output_path = "../../data/processed/creditcard_processed.csv"

    process(input_path,output_path)

