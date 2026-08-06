
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_inspect_data(file_path="ai4i2020.csv"):
    """
    Loads the raw telemetry matrix and inspects structural dimensions.
    Performs automated missing value verification to log data integrity.
    """


    # Loads the raw dataset and inspects structural dimensions.
    print(" Loading raw dataset...")
    
    # Ingest the telemetry csv file
    df = pd.read_csv(file_path)
    
    # Verify shape parameters 
    rows, cols = df.shape
    print(f" Dataset loaded successfully. Shape: {rows} rows, {cols} columns.")
    
    # Check for missing values across the dataset
    null_counts = df.isnull().sum().sum()
    print(f"Integrity Check: Identified {null_counts} total missing/null entries.")
    
    return df

def clean_and_select_features(df):
    """
    Step 2: Removes unique tracking IDs and target leakage columns.
    Isolates the core features from the binary Machine failure target.
    """
    print("\n[PROGRESS - DATA PIPELINE] Executing Feature Selection & Leakage Elimination...")
    
    # Isolate the target variable (Binary operational machine failure label)
    y = df['Machine failure'].copy()
    print(f" Isolated binary target variable 'Machine failure'. Failure Rate: {round(y.mean()*100, 2)}%")
    
    # Explicitly define columns to drop to prevent overfitting and data leakage
    # Dropping UDI/Product ID (Identifiers) and the 5 specific failure modes (Leakage sources)
    columns_to_drop = ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    X_raw = df.drop(columns=columns_to_drop)
    print(f" Dropped unique identifiers and target leakage columns: {columns_to_drop[2:]}")
    print(f" Extracted raw feature matrix shape: {X_raw.shape}")
    
    return X_raw, y

def preprocess_and_scale_features(X_raw):
    """
    Step 3: Converts categorical data via One-Hot Encoding.
    Applies standard feature scaling to continuous operational sensor metrics.
    """
    print("\n[PROGRESS - DATA PIPELINE] Initiating Feature Transformation & Preprocessing...")
    
    # 3.1 Separate features into categorical and numerical trackers
    categorical_cols = ['Type']
    numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 
                      'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    # 3.2 Apply One-Hot Encoding to the categorical asset quality Type (L, M, H)
    # Using drop_first=False to explicitly show columns for L, M, and H in final feature maps
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, dtype=int)
    print(" Applied One-Hot Encoding to categorical asset 'Type'.")
    print(f" Expanded columns check: {list(X_encoded.columns)}")
    
    # 3.3 Apply Feature Scaling (Standard Scaling) to all continuous operational variables
    print(" Initializing StandardScaler to balance multi-sensor metric ranges...")
    scaler = StandardScaler()
    
    # Fit and transform only the numerical columns
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
    print(" [SUCCESS] Applied Standard Scaling (Mean=0, Var=1) across continuous parameters:")
    print("  (Motor Winding Temperatures, Drive Shaft Pulley Physics, Cumulative Belt Wear)")
    print(f" [FINAL] Preprocessed feature matrix shape: {X_encoded.shape}\n")
    
    return X_encoded

def run_data_pipeline_module(file_path="ai4i2020.csv"):
    """
    Orchestration function to execute the full data engineering block sequentially.
    """
    print("="*60)
    print("Starting Preprocessing and Data Refinery Phase...")
    print("="*60)
    
    raw_df = load_and_inspect_data(file_path)
    X_raw, y = clean_and_select_features(raw_df)
    X_processed = preprocess_and_scale_features(X_raw)
    
    return X_processed, y

if __name__ == "__main__":
    # Allows the refinery script to be executed and tested independently in terminal
    X, y = run_data_pipeline_module()
