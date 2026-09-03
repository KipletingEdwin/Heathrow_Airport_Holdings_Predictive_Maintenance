
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


# Compute classification performance metrics and print a summary with absolute tallies.
def log_metrics(name, y_true, y_pred):
    # Build raw evaluation counts from index mappings
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    
    # Calculate performance percentages and format them for clean column logging
    print(f"{name:<10} | Acc: {accuracy_score(y_true, y_pred)*100:5.2f}% | "
          f"Prec: {precision_score(y_true, y_pred, zero_division=0)*100:5.2f}% | "
          f"Rec: {recall_score(y_true, y_pred, zero_division=0)*100:5.2f}% | "
          f"F1: {f1_score(y_true, y_pred, zero_division=0)*100:5.2f}%")
    print(f"           -> [TP: {tp} | FN: {fn} | FP: {fp} | TN: {tn}]")


# Generate and save a customized confusion matrix heatmap layout.
def plot_confusion(model, X_test_scaled, y_test, cmap, filename):
    # Predict outcomes and break results down into matrix metrics
    cm = confusion_matrix(y_test, model.predict(X_test_scaled), labels=[0, 1])
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    # Re-order variables manually to prioritize positive instances in top-left
    grid = np.array([[tp, fn],
                     [fp, tn]])
    labels = np.array([
        [f"True Positive (TP)\n{tp}", f"False Negative (FN)\n{fn}"],
        [f"False Positive (FP)\n{fp}", f"True Negative (TN)\n{tn}"],
    ])

    # Generate a visual plot window layout using Seaborn heatmaps
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(grid, annot=labels, fmt="", cmap=cmap, cbar=False,
                xticklabels=['Predicted Failure (1)', 'Predicted Healthy (0)'],
                yticklabels=['Actual Failure (1)', 'Actual Healthy (0)'])
    plt.ylabel('Actual conveyor state')
    plt.xlabel('Predicted state')
    plt.tight_layout()
    
    # Save chart to hard drive and free active drawing memory
    plt.savefig(filename, dpi=300)
    plt.close()


# Extract, log, and plot feature importance scores from the trained model.
def plot_feature_importance(model, feature_names, filename):
    # Align structural features with calculated internal model importance scores
    fi = (pd.DataFrame({'Feature': feature_names,
                        'Importance': model.feature_importances_})
          .sort_values(by='Importance', ascending=False))
    
    # Generate a horizontal bar plot showing driving elements ranking
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=fi,
                hue='Feature', legend=False, palette='viridis')
    plt.xlabel('Relative feature importance')
    plt.ylabel('Operational variable')
    plt.tight_layout()
    
    # Export graphic asset and free memory
    plt.savefig(filename, dpi=300)
    plt.close()
    
    # Print out raw textual importance readings for logs
    print("\nFeature importance ranking (XGBoost champion):")
    for _, row in fi.iterrows():
        print(f"   {row['Feature']:<24} {row['Importance']:.4f}")


# Orchestrate full end-to-end pipeline execution.
if __name__ == "__main__":
    # Stage 1: Ingest, inspect metrics, wipe noise, and break down sets
    df_raw = load_and_inspect_data()
    verify_target_distribution(df_raw)
    X, y = execute_feature_stripping(df_raw)
    X_train_scaled, X_test_scaled, y_train, y_test = partition_and_scale_data(X, y)

    # Stage 2: Establish base models without applying correction parameters
    rf_naive = RandomForestClassifier(random_state=42).fit(X_train_scaled, y_train)
    xgb_naive = xgb.XGBClassifier(random_state=42, eval_metric='logloss').fit(X_train_scaled, y_train)

    # Stage 3: Measure targets ratio and search cross-validation parameters
    weight = extract_imbalance_weight(y_train)
    rf_best, xgb_best = execute_grid_search_tuning(X_train_scaled, y_train, weight)

    # Stage 4: Train optimized champions using custom class weights
    rf_champ = RandomForestClassifier(**rf_best, class_weight='balanced',
                                      random_state=42).fit(X_train_scaled, y_train)
    xgb_champ = xgb.XGBClassifier(**xgb_best, scale_pos_weight=weight,
                                  random_state=42, eval_metric='logloss').fit(X_train_scaled, y_train)

    # Stage 5: Log final scores on out-of-sample data points
    print("\n" + "=" * 72)
    print("EXPERIMENTAL RESULTS: MODEL COMPARISON (test set)")
    print("=" * 72)
    log_metrics("RF Naive", y_test, rf_naive.predict(X_test_scaled))
    log_metrics("RF Champ", y_test, rf_champ.predict(X_test_scaled))
    print("-" * 72)
    log_metrics("XGB Naive", y_test, xgb_naive.predict(X_test_scaled))
    log_metrics("XGB Champ", y_test, xgb_champ.predict(X_test_scaled))
    print("=" * 72)

    # Stage 6: Build diagnostic visualization assets
    plot_confusion(rf_champ, X_test_scaled, y_test,
                   "Oranges", "figure_rf_confusion.png")
    plot_confusion(xgb_champ, X_test_scaled, y_test,
                   "Blues", "figure_xgb_confusion.png")
    plot_feature_importance(xgb_champ, list(X.columns),
                            "figure_feature_importance.png")

    print("\nDiagnostic plots saved to working directory.")
