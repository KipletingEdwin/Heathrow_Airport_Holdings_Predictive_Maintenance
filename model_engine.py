
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


def partition_and_scale_data(X, y):

    print("\n Partitioning using startification ")
    
    # Split data into 80/20 ratio, ensuring failure ratios are balanced both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    print(f" Row balance: {len(y_train)} training records vs {len(y_test)} validation records.")
    
    # Enforce Leakage-Free Feature Scaling using StandardScaler
    print("\n Feature Scaling Protocol...")
    scaler = StandardScaler()
    
    # Identify the continuous column names matching our updated naming pattern
    numerical_features = ['Air temperature _K', 'Process temperature _K', 
                          'Rotational speed _rpm', 'Torque _Nm', 'Tool wear _min']
    
    # FIT and TRANSFORM strictly on the 80% training rows
    X_train_scaled = X_train.copy()
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    print(" StandardScaler calculated rules (Mean/Var) strictly from Training Block data.")
    
    # TRANSFORM ONLY on the 20% test rows for Zero look-ahead distribution pollution)
    X_test_scaled = X_test.copy()
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    print(" Evaluated validation block scaled using extracted training rules.")
    
    return X_train_scaled, X_test_scaled, y_train, y_test



def extract_imbalance_weight(y_train):

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    weight_value = neg_count / pos_count
    print(f"\n Computed Skew Base: 1 Fault per {round(weight_value )} Safe States.")
    return weight_value



def execute_grid_search_tuning(X_train, y_train, scale_pos_weight_val):

    print("\n Initializing Hyperparameter Grid Optimization Space ")
    recall_scorer = make_scorer(recall_score)
    
    # Explicitly populated lists to clear syntax errors
    rf_estimators_list = [50, 100]
    rf_depth_list = [3, 5]
    rf_split_list = [2, 5]
    
    xgb_estimators_list = [50, 100]
    xgb_depth_list = [3, 5]
    xgb_lr_list = [0.05, 0.1]
    
    # Tune Random Forest Hyperparameters
    print(" Sweeping Random Forest Grid parameters via 3-Fold CV...")
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_grid = {
        'n_estimators': rf_estimators_list, 
        'max_depth': rf_depth_list, 
        'min_samples_split': rf_split_list
    }
    rf_search = GridSearchCV(estimator=rf_base, param_grid=rf_grid, scoring=recall_scorer, cv=3)
    rf_search.fit(X_train, y_train)
    print(f" Extracted Optimal RF Structure: {rf_search.best_params_}")
    
    # Tune XGBoost Hyperparameters
    print(" Sweeping XGBoost Grid parameters via 3-Fold CV...")
    xgb_base = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight_val, random_state=42, eval_metric='logloss')
    xgb_grid = {
        'n_estimators': xgb_estimators_list, 
        'max_depth': xgb_depth_list, 
        'learning_rate': xgb_lr_list
    }
    xgb_search = GridSearchCV(estimator=xgb_base, param_grid=xgb_grid, scoring=recall_scorer, cv=3)
    xgb_search.fit(X_train, y_train)
    print(f" Extracted Optimal XGBoost Structure: {xgb_search.best_params_}")
    
    return rf_search.best_params_, xgb_search.best_params_



def run_modeling_engine_module(X, y):

    X_train, X_test, y_train, y_test = partition_and_scale_data(X, y)
    xgb_weight = extract_imbalance_weight(y_train)
    rf_best_params, xgb_best_params = execute_grid_search_tuning(X_train, y_train, xgb_weight)
    
    return rf_best_params, xgb_best_params, X_train, X_test, y_train, y_test




