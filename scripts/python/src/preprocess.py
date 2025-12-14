"""
Preprocess data for credit card default prediction.
Matches feature engineering from Rust project.

Usage: python src/preprocess.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Feature groups (matching Rust project)
NUMERIC_FEATURES = [
    'LIMIT_BAL', 'AGE',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
]

CATEGORICAL_FEATURES = ['SEX', 'EDUCATION', 'MARRIAGE']

PAY_COLS = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']


def load_data(path="data/UCI_Credit_Card.csv"):
    """Load and clean the dataset."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    
    # Rename columns to match expected format
    df = df.drop(columns=['ID'])
    df = df.rename(columns={
        'default.payment.next.month': 'default',
        'PAY_0': 'PAY_1'  # Rename to be consistent
    })
    
    # Clean categorical values
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
    
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def engineer_features(df):
    """
    Add engineered features matching Rust project.
    These features improve model performance for fraud detection.
    """
    df = df.copy()
    
    # --- Payment Behavior Features ---
    # Average payment delay
    df['avg_pay_delay'] = df[PAY_COLS].mean(axis=1)
    
    # Maximum payment delay
    df['max_pay_delay'] = df[PAY_COLS].max(axis=1)
    
    # Number of months with delay (payment > 0)
    df['months_delayed'] = (df[PAY_COLS] > 0).sum(axis=1)
    
    # Payment trend (improving or worsening)
    df['pay_trend'] = df['PAY_1'] - df['PAY_6']
    
    # --- Bill Amount Features ---
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    
    # Average bill amount
    df['avg_bill_amt'] = df[bill_cols].mean(axis=1)
    
    # Bill trend (increasing or decreasing)
    df['bill_trend'] = df['BILL_AMT1'] - df['BILL_AMT6']
    
    # Bill volatility
    df['bill_volatility'] = df[bill_cols].std(axis=1)
    
    # --- Payment Amount Features ---
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    # Average payment amount
    df['avg_pay_amt'] = df[pay_amt_cols].mean(axis=1)
    
    # Total payment amount
    df['total_pay_amt'] = df[pay_amt_cols].sum(axis=1)
    
    # --- Utilization Features ---
    # Credit utilization ratio
    df['utilization_ratio'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)
    
    # Over limit indicator
    df['over_limit'] = (df['BILL_AMT1'] > df['LIMIT_BAL']).astype(int)
    
    # Payment to bill ratio
    df['pay_bill_ratio'] = df['PAY_AMT1'] / (df['BILL_AMT1'].abs() + 1)
    
    # --- Demographic Features ---
    # Age groups (0-4)
    df['age_group'] = pd.cut(
        df['AGE'],
        bins=[0, 25, 35, 45, 55, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(2)
    
    return df


def get_feature_names():
    """Get all feature names after engineering."""
    engineered = [
        'avg_pay_delay', 'max_pay_delay', 'months_delayed', 'pay_trend',
        'avg_bill_amt', 'bill_trend', 'bill_volatility',
        'avg_pay_amt', 'total_pay_amt',
        'utilization_ratio', 'over_limit', 'pay_bill_ratio', 'age_group'
    ]
    return NUMERIC_FEATURES + PAY_COLS + CATEGORICAL_FEATURES + engineered


def main():
    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Split data
    print("\nSplitting data (80/20)...")
    X = df.drop(columns=['default'])
    y = df['default']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)} records")
    print(f"Test: {len(X_test)} records")
    print(f"Default rate (train): {y_train.mean():.1%}")
    print(f"Default rate (test): {y_test.mean():.1%}")
    
    # Engineer features
    print("\nEngineering features...")
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)
    
    print(f"Features after engineering: {len(X_train.columns)}")
    
    # Save processed data
    os.makedirs("results", exist_ok=True)
    X_train.to_csv("results/X_train.csv", index=False)
    X_test.to_csv("results/X_test.csv", index=False)
    y_train.to_csv("results/y_train.csv", index=False, header=['default'])
    y_test.to_csv("results/y_test.csv", index=False, header=['default'])
    
    # Save feature names
    feature_names = list(X_train.columns)
    with open("results/feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))
    
    print(f"\n✓ Saved: X_train ({X_train.shape}), X_test ({X_test.shape})")
    print(f"✓ Features: {len(feature_names)}")


if __name__ == "__main__":
    main()
