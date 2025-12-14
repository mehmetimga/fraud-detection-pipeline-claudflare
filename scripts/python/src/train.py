"""
Train models for credit card default prediction.
Usage: python src/train.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle

# Feature groups
NUMERIC_FEATURES = ['LIMIT_BAL', 'AGE', 
                    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
                    'avg_bill', 'avg_payment', 'payment_ratio', 'credit_util']
ORDINAL_FEATURES = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'times_late', 'max_months_late']
CATEGORICAL_FEATURES = ['SEX', 'EDUCATION', 'MARRIAGE']

def get_preprocessor():
    """Create preprocessing pipeline."""
    return ColumnTransformer([
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('ord', 'passthrough', ORDINAL_FEATURES),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), CATEGORICAL_FEATURES)
    ])

def main():
    print("Loading processed data...")
    X_train = pd.read_csv("results/X_train.csv")
    y_train = pd.read_csv("results/y_train.csv").values.ravel()
    
    print("Creating pipeline...")
    preprocessor = get_preprocessor()
    pipe = make_pipeline(preprocessor, GradientBoostingClassifier(random_state=42))
    
    print("Tuning hyperparameters...")
    param_dist = {
        'gradientboostingclassifier__n_estimators': [50, 100, 200],
        'gradientboostingclassifier__learning_rate': [0.01, 0.1, 0.2],
        'gradientboostingclassifier__max_depth': [3, 5, 7],
    }
    
    search = RandomizedSearchCV(pipe, param_dist, n_iter=10, cv=3, scoring='f1', random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    
    print(f"Best F1: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    
    # Save model
    with open("results/best_model.pkl", "wb") as f:
        pickle.dump(search.best_estimator_, f)
    
    print("✓ Model saved to results/best_model.pkl")

if __name__ == "__main__":
    main()

