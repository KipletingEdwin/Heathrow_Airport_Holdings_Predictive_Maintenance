
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


# Split data into stratified train/test sets and apply feature scaling to continuous columns.
def partition_and_scale_data(X, y):
    print("\nPartitioning via stratified 80/20 split...")
    # Use stratification to ensure train and test sets have matching failure ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Row balance: {len(y_train)} train vs {len(y_test)} test records.")
    print(f"Train failures: {int(sum(y_train))} / {len(y_train)}  |  "
          f"Test failures: {int(sum(y_test))} / {len(y_test)}")

    # Isolate continuous variables by ignoring the categorical 'Type' index
    numerical_features = [c for c in X.columns if c != 'Type']

    print("\nApplying split-before-scale standardisation...")
    scaler = StandardScaler()

    # Calculate transformation rules purely from training data to avoid leakage
    X_train_scaled = X_train.copy()
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])

    # Project the pre-calculated training weights onto the unseen testing variables
    X_test_scaled = X_test.copy()
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    print("Scaler fitted on training rows only; test rows transformed with training statistics.")

    return X_train_scaled, X_test_scaled, y_train, y_test


# Calculate the ratio of negative instances to positive instances for class weighting.
def extract_imbalance_weight(y_train):
    # Find total instances of non-failures versus failures in the training batch
    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))

    # Divide healthy instances by failures to determine the structural class multiplier
    weight_value = neg_count / pos_count
    print(f"\nCost-sensitive weight (neg/pos): {weight_value:.2f}  "
          f"({neg_count} normal / {pos_count} failure)")
    return weight_value


# Run a 3-fold cross-validated grid search tuning over Random Forest and XGBoost targeting optimal F1.
def execute_grid_search_tuning(X_train_scaled, y_train, scale_pos_weight_val):
    print("\nRunning hyperparameter grid search (3-fold CV, optimised for F1)...")

    # Set up F1 scoring metric to optimize both precision and recall evenly
    f1_scorer = make_scorer(f1_score)

    # Initialize a baseline Random Forest that automatically adjusts weights
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
    }
    # Run the grid search cross-validation loop for Random Forest
    rf_search = GridSearchCV(rf_base, rf_grid, scoring=f1_scorer, cv=3)
    rf_search.fit(X_train_scaled, y_train)
    print(f"RF  best CV F1: {rf_search.best_score_:.4f} | params: {rf_search.best_params_}")

    # Initialize XGBoost and pass the custom positive class multiplier parameter
    xgb_base = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight_val,
        random_state=42, eval_metric='logloss'
    )
    xgb_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
    }
    # Run the grid search cross-validation loop for XGBoost
    xgb_search = GridSearchCV(xgb_base, xgb_grid, scoring=f1_scorer, cv=3)
    xgb_search.fit(X_train_scaled, y_train)
    print(f"XGB best CV F1: {xgb_search.best_score_:.4f} | params: {xgb_search.best_params_}")

    # Return the configurations that scored the best F1 results
    return rf_search.best_params_, xgb_search.best_params_







































