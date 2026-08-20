
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# Core pipeline imports from your workspace modules
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


def log_metrics(name, y_true, y_pred):
    # Using labels=[1,0] to ensure terminal string counts print in alignment with the inverted visual charts
    cm_inverted = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn, fp, tn = cm_inverted.ravel()
    print(f"{name:<10} | Acc: {accuracy_score(y_true, y_pred)*100:.2f}% | Prec: {precision_score(y_true, y_pred)*100:.2f}% | Rec: {recall_score(y_true, y_pred)*100:.2f}% | F1: {f1_score(y_true, y_pred)*100:.2f}%")
    print(f"           -> Grid Counts: [TP: {tp:<4} | FN: {fn:<3} | FP: {fp:<2} | TN: {tn}]")

def export_all_thesis_plots(rf_model, xgb_model, X_test, y_test, feature_names):
    sns.set_theme(style="whitegrid")
    
    # 1. GENERATE INVERTED RANDOM FOREST CONFUSION MATRIX (TP ON TOP-LEFT)
    rf_cm_inverted = confusion_matrix(y_test, rf_model.predict(X_test), labels=[1, 0])
    rtp, rfn, rfp, rtn = rf_cm_inverted.ravel()
    rf_labels = np.array([
        [f"True Positive (TP)\n{rtp}", f"False Negative (FN)\n{rfn}"],
        [f"False Positive (FP)\n{rfp}", f"True Negative (TN)\n{rtn}"]
    ])
    
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(rf_cm_inverted, annot=rf_labels, fmt="", cmap='Oranges', cbar=False,
                xticklabels=['Predicted Failure (1)', 'Predicted Healthy (0)'], 
                yticklabels=['Actual Failure (1)', 'Actual Healthy (0)'])
    plt.title('Figure 6.1a: Random Forest Champion Confusion Matrix Grid')
    plt.ylabel('Ground Truth (Actual Conveyor State)')
    plt.xlabel('Algorithmic Decision Profile')
    plt.tight_layout()
    plt.savefig('figure_6_1a_rf_confusion_matrix.png', dpi=300)
    plt.close()

    # 2. GENERATE INVERTED XGBOOST CHAMPION CONFUSION MATRIX (TP ON TOP-LEFT)
    xgb_cm_inverted = confusion_matrix(y_test, xgb_model.predict(X_test), labels=[1, 0])
    xtp, xfn, xfp, xtn = xgb_cm_inverted.ravel()
    xgb_labels = np.array([
        [f"True Positive (TP)\n{xtp}", f"False Negative (FN)\n{xfn}"],
        [f"False Positive (FP)\n{xfp}", f"True Negative (TN)\n{xtn}"]
    ])
    
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(xgb_cm_inverted, annot=xgb_labels, fmt="", cmap='Blues', cbar=False,
                xticklabels=['Predicted Failure (1)', 'Predicted Healthy (0)'], 
                yticklabels=['Actual Failure (1)', 'Actual Healthy (0)'])
    plt.title('Figure 6.1b: XGBoost Champion Confusion Matrix Grid')
    plt.ylabel('Ground Truth (Actual Conveyor State)')
    plt.xlabel('Algorithmic Decision Profile')
    plt.tight_layout()
    plt.savefig('figure_6_1b_xgb_confusion_matrix.png', dpi=300)
    plt.close()

    # 3. RENDER FIGURE 7.1: FEATURE IMPORTANCE BAR CHART (XGBOOST SENSOR RANKINGS)
    fi_df = pd.DataFrame({'Sensor Metric': feature_names, 'Importance': xgb_model.feature_importances_}).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Sensor Metric', data=fi_df, palette='viridis', hue='Sensor Metric', legend=False)
    plt.title('Figure 7.1: XGBoost Champion Sensor Importance Rankings')
    plt.xlabel('Relative Feature Gain Weight (Information Content)')
    plt.ylabel('Conveyor Operational Variable')
    plt.tight_layout()
    plt.savefig('figure_7_1_feature_importance.png', dpi=300)
    plt.close()
    
    print("\n[SUCCESS] All 3 visual assets saved completely to your workspace with TP on the top-left:")
    print(" -> 'figure_6_1a_rf_confusion_matrix.png'")
    print(" -> 'figure_6_1b_xgb_confusion_matrix.png'")
    print(" -> 'figure_7_1_feature_importance.png'\n")

if __name__ == "__main__":
    # 1. Data Refinery Pipeline Phase
    df_raw = load_and_inspect_data()
    verify_target_distribution(df_raw)
    X_processed, y_target = execute_feature_stripping(df_raw)
    X_train, X_test, y_train, y_test = partition_and_scale_data(X_processed, y_target)
    
    # 2. Naive Baseline Modeling Phase (Control Group)
    rf_naive = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    xgb_naive = xgb.XGBClassifier(random_state=42, eval_metric='logloss').fit(X_train, y_train)
    
    # 3. Hyperparameter Grid Space Tuning Phase
    xgb_weight = extract_imbalance_weight(y_train)
    rf_best, xgb_best = execute_grid_search_tuning(X_train, y_train, xgb_weight)
    
    # 4. Upgraded Cost-Sensitive Champion Phase
    rf_champ = RandomForestClassifier(**rf_best, class_weight='balanced', random_state=42).fit(X_train, y_train)
    xgb_champ = xgb.XGBClassifier(**xgb_best, scale_pos_weight=xgb_weight, random_state=42, eval_metric='logloss').fit(X_train, y_train)
    
    # 5. Core Metric Progression Displays
    print("\n" + "="*70 + "\nEXPERIMENTAL PROGRESS REPORT: MATRIX COMPARISONS\n" + "="*70)
    log_metrics("RF Naive", y_test, rf_naive.predict(X_test))
    log_metrics("RF Champ", y_test, rf_champ.predict(X_test))
    print("-"*70)
    log_metrics("XGB Naive", y_test, xgb_naive.predict(X_test))
    log_metrics("XGB Champ", y_test, xgb_champ.predict(X_test))
    print("="*70)
    
    # 6. Run Chart Generator for BOTH models (TP on top-left)
    export_all_thesis_plots(rf_champ, xgb_champ, X_test, y_test, list(X_processed.columns))
