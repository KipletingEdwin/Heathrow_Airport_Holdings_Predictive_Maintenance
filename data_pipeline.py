
import pandas as pd


# Load the source CSV dataset and evaluate its basic structural dimensions.
def load_and_inspect_data(file_path="ai4i2020.csv"):
    print("Loading raw dataset...")
    try:
        # Attempt to read the comma-separated data file
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Intercept missing file errors to output a clear workspace setup hint
        raise FileNotFoundError(
            f"Could not find '{file_path}'. Place ai4i2020.csv in the same folder as this script."
        )

    # Capture the row and column counts for verification
    rows, cols = df.shape
    print(f"Dataset shape: {rows} rows, {cols} columns.")

    # Check for empty cells or missing observations across the whole dataframe
    null_entries = df.isnull().sum().sum()
    print(f"Data check: {null_entries} missing entries identified.")
    return df


# Analyze the distribution of the binary target variable to quantify class imbalance.
def verify_target_distribution(df):
    print("\nVerifying target distribution...")
    # Tally instances belonging to healthy (0) versus failed (1) machinery
    counts = df['Machine failure'].value_counts()
    normal_count = int(counts.get(0, 0))
    failure_count = int(counts.get(1, 0))
    total = len(df)

    print(f"Baseline: {normal_count} healthy vs {failure_count} failure samples.")
    # Calculate percentages to see how skewed the data is before modeling
    print(f"Imbalance: {((failure_count / total) * 100):.2f}% failure rate.")


# Clean dataset: remove label noise, isolate target, drop leakage variables, map ranks, and clean headers.
def execute_feature_stripping(df):
    # Strip random-failure noise records where no physical fault signature exists
    df_cleaned = df[df['RNF'] == 0].copy()
    print(f"\nRows after RNF drop: {len(df_cleaned)} "
          f"(removed {len(df) - len(df_cleaned)} random-failure rows)")

    # Separate the target outcome variable from the predictor fields
    y = df_cleaned['Machine failure'].copy()

    # Define columns that cause data leakage or contain redundant identifiers
    drop_columns = ['UDI', 'Product ID', 'Machine failure',
                    'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    X_raw = df_cleaned.drop(columns=drop_columns)

    # Convert low, medium, and high quality type tiers into numeric rankings
    capacity_map = {'L': 1, 'M': 2, 'H': 3}
    X_raw['Type'] = X_raw['Type'].map(capacity_map)
    print("Encoded asset quality tiers: L->1, M->2, H->3")

    # Replace mathematical symbols and brackets to ensure XGBoost features are valid
    X_raw.columns = (X_raw.columns
                     .str.replace('[', '_', regex=False)
                     .str.replace(']', '', regex=False)
                     .str.replace('<', '', regex=False))

    print(f"Features prepared for training: {list(X_raw.columns)}")
    return X_raw, y
