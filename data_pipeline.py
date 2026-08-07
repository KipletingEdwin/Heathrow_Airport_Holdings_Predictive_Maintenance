
# import numpy as np
import pandas as pd


def load_and_inspect_data(file_path="ai4i2020.csv"):

    print(" Loading raw dataset...")
    
    df = pd.read_csv(file_path)
    
    # Checking shape parameters 
    rows, cols = df.shape
    print(f" Dataset Shape: {rows} rows, {cols} columns.")
    
    # Check if there are any mising values
    null_entries = df.isnull().sum().sum()
    print(f" Data Check: Identified {null_entries} missing entries.")
    
    return df

def verify_target_distribution(df):
    print("\n Verifying target distribution")

    counts = df['Machine failure'].value_counts()
    normal_count = counts.get(0, 0) # If 0 is not found, return 0
    failure_count = counts.get(1, 0) #If 1 is not found, return 0
    total = len(df)
    
    print(f" Baseline : {normal_count} Healthy Samples vs {failure_count} Failure Samples.")
    print(f" Imbalance Ratio: {round((failure_count/total)*100, 2)}% Breakdown Rate.")

def execute_feature_stripping(df):

    print("\n Enforcing Feature Selection & Leakage Prevention Protocol...")
    
    # Isolate target variable
    y = df['Machine failure'].copy()
    
    # Build list of attributes to drop to eliminate look-ahead data pollution
    drop_columns = ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    X_raw = df.drop(columns=drop_columns)
    
    print(f" Modified dataset shape: {X_raw.shape}")
    
    # Step 5: Separate feature columns conceptually for encoding and scaling downstream
    categorical_cols = ['Type']
    # numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 
    #                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    # Apply One-Hot Encoding to categorical asset quality Type (L, M, H)
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, dtype=int)
    print(" Applied One-Hot Encoding transformation to categorical attribute 'Type'.")
    
    # Reformat columns to remove brackets to protect downstream XGBoost engine tracking strings
    X_encoded.columns = (X_encoded.columns
                         .str.replace('[', '_', regex=False)
                         .str.replace(']', '', regex=False)
                         .str.replace('<', '', regex=False))
    
    print(f" Refactored feature column names: {list(X_encoded.columns)}")
    
    return X_encoded, y



# if __name__ == "__main__":
#     # 1. Load and inspect the dataset
#     df_raw = load_and_inspect_data()
    
#     # 2. Check the class imbalance
#     verify_target_distribution(df_raw)
    
#     # 3. Strip features and encode
#     X, y = execute_feature_stripping(df_raw)
    
#     print("\n="*60)
#     print(" Data Pipeline Execution Completed Successfully!")
#     print("="*60)





