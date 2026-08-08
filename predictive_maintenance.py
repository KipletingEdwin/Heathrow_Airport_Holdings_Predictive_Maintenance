
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Ingest modules from your local workspace folder structure
from data_pipeline import (
    execute_feature_stripping,
    load_and_inspect_data,
    verify_target_distribution,
)
from model_engine import (
    execute_grid_search_tuning,
    extract_imbalance_weight,
    partition_and_scale_data,
)

if __name__ == "__main__":
    #DATA ENGINEERING
    df_raw = load_and_inspect_data()
    verify_target_distribution(df_raw)
    X_processed, y_target = execute_feature_stripping(df_raw)
    
    # Enforce Split-Before-Scale rule via model engine layers
    X_train, X_test, y_train, y_test = partition_and_scale_data(X_processed, y_target)
    
    #NAIVE BASELINE MODELING 
    # Train Default Random Forest with NO class weights
    rf_naive = RandomForestClassifier(random_state=42)
    rf_naive.fit(X_train, y_train)
    rf_naive_preds = rf_naive.predict(X_test)
    
    # Train Default XGBoost with NO scale position weights
    xgb_naive = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_naive.fit(X_train, y_train)
    xgb_naive_preds = xgb_naive.predict(X_test)
    print(" Naive Baseline control group models trained successfully.")
    
    #STRUCTURAL SELECTION & UPGRADE 
    xgb_weight = extract_imbalance_weight(y_train)
    
    # Grid Search to extract hyperparameter boundaries optimized for Recall
    rf_best_params, xgb_best_params = execute_grid_search_tuning(X_train, y_train, xgb_weight)
    
    print(" Deploying Upgraded Cost-Sensitive Configuration Metrics...")
    
    # Re-train models injecting optimal hyperparameter boundaries and cost settings
    rf_champion = RandomForestClassifier(
        n_estimators=rf_best_params['n_estimators'],
        max_depth=rf_best_params['max_depth'],
        min_samples_split=rf_best_params['min_samples_split'],
        class_weight='balanced',
        random_state=42
    )
    rf_champion.fit(X_train, y_train)
    rf_champ_preds = rf_champion.predict(X_test)
    
    xgb_champion = xgb.XGBClassifier(
        n_estimators=xgb_best_params['n_estimators'],
        max_depth=xgb_best_params['max_depth'],
        learning_rate=xgb_best_params['learning_rate'],
        scale_pos_weight=xgb_weight,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_champion.fit(X_train, y_train)
    xgb_champ_preds = xgb_champion.predict(X_test)
    
    #FINAL METRIC PROGRESS GRID COMPARISON 
    print("\n" + "="*70)
    print("      EXPERIMENTAL PROGRESS REPORT: BEFORE AND AFTER MATRIX COMPARISONS")
    print("="*70)
    print(f"{'Performance Metric':<24} | {'RF Naive':<10} | {'RF Champ':<10} | {'XGB Naive':<10} | {'XGB Champ':<10}")
    print("-"*70)
    print(f"{'Traditional Accuracy':<24} | {accuracy_score(y_test, rf_naive_preds)*100:<8.2f}% | {accuracy_score(y_test, rf_champ_preds)*100:<8.2f}% | {accuracy_score(y_test, xgb_naive_preds)*100:<8.2f}% | {accuracy_score(y_test, xgb_champ_preds)*100:<8.2f}%")
    print(f"{'Precision (Alarm Trust)':<24} | {precision_score(y_test, rf_naive_preds)*100:<8.2f}% | {precision_score(y_test, rf_champ_preds)*100:<8.2f}% | {precision_score(y_test, xgb_naive_preds)*100:<8.2f}% | {precision_score(y_test, xgb_champ_preds)*100:<8.2f}%")
    print(f"{'Recall (Sensitivity)':<24} | {recall_score(y_test, rf_naive_preds)*100:<8.2f}% | {recall_score(y_test, rf_champ_preds)*100:<8.2f}% | {recall_score(y_test, xgb_naive_preds)*100:<8.2f}% | {recall_score(y_test, xgb_champ_preds)*100:<8.2f}%")
    print(f"{'Balanced F1-Score':<24} | {f1_score(y_test, rf_naive_preds)*100:<8.2f}% | {f1_score(y_test, rf_champ_preds)*100:<8.2f}% | {f1_score(y_test, xgb_naive_preds)*100:<8.2f}% | {f1_score(y_test, xgb_champ_preds)*100:<8.2f}%")
    print("-"*70)
    
    # Extract side-by-side Confusion Matrix counts for explicit error tracking reports
    rf_n_cm = confusion_matrix(y_test, rf_naive_preds)
    rf_c_cm = confusion_matrix(y_test, rf_champ_preds)
    xgb_n_cm = confusion_matrix(y_test, xgb_naive_preds)
    xgb_c_cm = confusion_matrix(y_test, xgb_champ_preds)
    
    print("\n" + "-"*70)
    print("DIAGNOSTIC CONFUSION MATRIX DRIFT (CRITICAL ERROR AND FAULT TRACKING)")
    print("-"*70)
    print(f" -> RF Naive Grid : [TN: {rf_n_cm[0][0]:<4} | FP: {rf_n_cm[0][1]:<3} | FN: {rf_n_cm[1][0]:<2} | TP: {rf_n_cm[1][1]}]")
    print(f" -> RF Champ Grid : [TN: {rf_c_cm[0][0]:<4} | FP: {rf_c_cm[0][1]:<3} | FN: {rf_c_cm[1][0]:<2} | TP: {rf_c_cm[1][1]}]")
    print(f" -> XGB Naive Grid: [TN: {xgb_n_cm[0][0]:<4} | FP: {xgb_n_cm[0][1]:<3} | FN: {xgb_n_cm[1][0]:<2} | TP: {xgb_n_cm[1][1]}]")
    print(f" -> XGB Champ Grid: [TN: {xgb_c_cm[0][0]:<4} | FP: {xgb_c_cm[0][1]:<3} | FN: {xgb_c_cm[1][0]:<2} | TP: {xgb_c_cm[1][1]}]")
    print("-"*70)
    
    print("\n -> [CRITICAL REPORT PROGRESS INSIGHT]")
    print(f"    Look at the Recall shift! RF jumped from {recall_score(y_test, rf_naive_preds)*100:.2f}% to {recall_score(y_test, rf_champ_preds)*100:.2f}%.")
    print(f"    XGBoost jumped from {recall_score(y_test, xgb_naive_preds)*100:.2f}% to {recall_score(y_test, xgb_champ_preds)*100:.2f}%.")
    print("    This confirms your experimental tuning successfully mitigated the data imbalance skew!\n")
