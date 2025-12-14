"""
Evaluate model on test set.
Usage: python src/evaluate.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import pickle

def main():
    print("Loading data and model...")
    X_test = pd.read_csv("results/X_test.csv")
    y_test = pd.read_csv("results/y_test.csv").values.ravel()
    
    with open("results/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
    
    test_f1 = f1_score(y_test, y_pred)
    test_roc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"F1 Score:  {test_f1:.4f}")
    print(f"ROC-AUC:   {test_roc:.4f}")
    print("="*50)
    
    # Save results
    results = pd.DataFrame({
        'Metric': ['F1 Score', 'ROC-AUC'],
        'Value': [test_f1, test_roc]
    })
    results.to_csv("results/test_results.csv", index=False)
    print("✓ Results saved to results/test_results.csv")

if __name__ == "__main__":
    main()

