"""
Evaluate all models on test set with ensemble comparison.
Matches evaluation from Rust project.

Usage: python src/evaluate.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, precision_score, recall_score,
    accuracy_score, precision_recall_curve
)
import joblib
import json
import os


# Model weights for ensemble (from R analysis)
MODEL_WEIGHTS = {
    'xgboost': 0.30,      # Best ROC-AUC
    'catboost': 0.28,     # Second best
    'lightgbm': 0.25,     # Third
    'random_forest': 0.17  # Fourth
}

# Optimal thresholds from R analysis
OPTIMAL_THRESHOLDS = {
    'xgboost': 0.61,
    'catboost': 0.57,
    'lightgbm': 0.60,
    'random_forest': 0.33,
    'ensemble': 0.56
}


def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """Find optimal classification threshold for F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Find best threshold
    best_idx = np.argmax(f1_scores[:-1])  # Exclude last element
    return thresholds[best_idx], f1_scores[best_idx]


def evaluate_model(y_true, y_pred, y_proba, model_name):
    """Evaluate a single model and return metrics."""
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'fraud_detection_rate': recall_score(y_true, y_pred),
        'false_alarm_rate': 1 - precision_score(y_true, y_pred)
    }
    return metrics


def ensemble_predict(models, X, weights=None):
    """
    Create ensemble predictions using weighted average.
    
    Args:
        models: Dict of model_name -> model
        X: Feature matrix
        weights: Dict of model_name -> weight (defaults to MODEL_WEIGHTS)
    
    Returns:
        Ensemble probabilities
    """
    if weights is None:
        weights = MODEL_WEIGHTS
    
    weighted_sum = np.zeros(len(X))
    total_weight = 0
    
    for name, model in models.items():
        if name in weights:
            proba = model.predict_proba(X)[:, 1]
            weighted_sum += proba * weights[name]
            total_weight += weights[name]
    
    return weighted_sum / total_weight


def print_comparison_table(results_df):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"\n{'Model':<15} {'ROC-AUC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'Accuracy':>10}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        print(f"{row['model']:<15} {row['roc_auc']:>10.4f} {row['f1']:>10.4f} "
              f"{row['precision']:>10.4f} {row['recall']:>10.4f} {row['accuracy']:>10.4f}")
    
    print("=" * 80)


def main():
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Load test data
    print("\nLoading test data...")
    X_test = pd.read_csv("results/X_test.csv")
    y_test = pd.read_csv("results/y_test.csv").values.ravel()
    
    print(f"Test samples: {len(X_test)}")
    print(f"Default rate: {y_test.mean():.1%}")
    
    # Load all models
    print("\nLoading models...")
    models = {}
    model_names = ['xgboost', 'catboost', 'lightgbm', 'random_forest']
    
    for name in model_names:
        model_path = f"results/{name}_model.pkl"
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
            print(f"  ✓ Loaded {name}")
        else:
            print(f"  ✗ Not found: {model_path}")
    
    if not models:
        print("No models found! Run train.py first.")
        return
    
    # Evaluate each model
    print("\n" + "=" * 60)
    print("INDIVIDUAL MODEL RESULTS")
    print("=" * 60)
    
    results = []
    all_probas = {}
    
    for name, model in models.items():
        # Get predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        all_probas[name] = y_proba
        
        # Use optimal threshold from R analysis
        threshold = OPTIMAL_THRESHOLDS.get(name, 0.5)
        y_pred = (y_proba >= threshold).astype(int)
        
        # Find optimal threshold for this data
        opt_threshold, opt_f1 = find_optimal_threshold(y_test, y_proba)
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, y_proba, name)
        metrics['optimal_threshold'] = opt_threshold
        metrics['threshold_used'] = threshold
        results.append(metrics)
        
        print(f"\n--- {name.upper()} ---")
        print(f"  Threshold used: {threshold:.2f}")
        print(f"  Optimal threshold: {opt_threshold:.2f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Ensemble evaluation
    print("\n" + "=" * 60)
    print("ENSEMBLE RESULTS")
    print("=" * 60)
    
    ensemble_proba = ensemble_predict(models, X_test)
    ensemble_threshold = OPTIMAL_THRESHOLDS['ensemble']
    ensemble_pred = (ensemble_proba >= ensemble_threshold).astype(int)
    
    # Find optimal ensemble threshold
    opt_threshold, opt_f1 = find_optimal_threshold(y_test, ensemble_proba)
    
    ensemble_metrics = evaluate_model(y_test, ensemble_pred, ensemble_proba, 'ensemble')
    ensemble_metrics['optimal_threshold'] = opt_threshold
    ensemble_metrics['threshold_used'] = ensemble_threshold
    results.append(ensemble_metrics)
    
    print(f"\n  Weights: {MODEL_WEIGHTS}")
    print(f"  Threshold used: {ensemble_threshold:.2f}")
    print(f"  Optimal threshold: {opt_threshold:.2f}")
    print(f"  Accuracy: {ensemble_metrics['accuracy']:.4f}")
    print(f"  Precision: {ensemble_metrics['precision']:.4f}")
    print(f"  Recall: {ensemble_metrics['recall']:.4f}")
    print(f"  F1 Score: {ensemble_metrics['f1']:.4f}")
    print(f"  ROC-AUC: {ensemble_metrics['roc_auc']:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('roc_auc', ascending=False)
    
    # Print comparison table
    print_comparison_table(results_df)
    
    # XGBoost vs CatBoost comparison
    print("\n" + "=" * 60)
    print("XGBOOST vs CATBOOST COMPARISON")
    print("=" * 60)
    
    xgb_metrics = results_df[results_df['model'] == 'xgboost'].iloc[0]
    cb_metrics = results_df[results_df['model'] == 'catboost'].iloc[0]
    
    print(f"\n{'Metric':<20} {'XGBoost':>12} {'CatBoost':>12} {'Difference':>12}")
    print("-" * 60)
    for metric in ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']:
        xgb_val = xgb_metrics[metric]
        cb_val = cb_metrics[metric]
        diff = xgb_val - cb_val
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        print(f"{metric:<20} {xgb_val:>12.4f} {cb_val:>12.4f} {diff_str:>12}")
    
    winner = "XGBoost" if xgb_metrics['roc_auc'] > cb_metrics['roc_auc'] else "CatBoost"
    print(f"\nWinner (by ROC-AUC): {winner}")
    
    # Model ranking
    print("\n" + "=" * 60)
    print("MODEL RANKING (by ROC-AUC)")
    print("=" * 60)
    
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"\n{emoji} #{i} {row['model'].upper()}")
        print(f"   ROC-AUC: {row['roc_auc']:.4f}")
        print(f"   F1: {row['f1']:.4f}")
    
    # Save results
    print("\n--- Saving Results ---")
    
    results_df.to_csv("results/model_comparison.csv", index=False)
    print("✓ Saved: results/model_comparison.csv")
    
    # Save optimal thresholds
    thresholds_df = results_df[['model', 'optimal_threshold', 'threshold_used']]
    thresholds_df.to_csv("results/optimal_thresholds.csv", index=False)
    print("✓ Saved: results/optimal_thresholds.csv")
    
    # Save predictions for analysis
    predictions_df = pd.DataFrame({
        'y_true': y_test,
        'xgboost_proba': all_probas.get('xgboost', np.zeros(len(y_test))),
        'catboost_proba': all_probas.get('catboost', np.zeros(len(y_test))),
        'lightgbm_proba': all_probas.get('lightgbm', np.zeros(len(y_test))),
        'random_forest_proba': all_probas.get('random_forest', np.zeros(len(y_test))),
        'ensemble_proba': ensemble_proba
    })
    predictions_df.to_csv("results/test_predictions.csv", index=False)
    print("✓ Saved: results/test_predictions.csv")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
