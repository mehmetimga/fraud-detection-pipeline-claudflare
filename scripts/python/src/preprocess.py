"""
Preprocess data for credit card default prediction.
Usage: python src/preprocess.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os

def load_data(path="data/UCI_Credit_Card.csv"):
    """Load and clean the dataset."""
    df = pd.read_csv(path)
    df = df.drop(columns=['ID'])
    df = df.rename(columns={'default.payment.next.month': 'default', 'PAY_0': 'PAY_1'})
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
    return df

def add_features(df):
    """Add engineered features."""
    df = df.copy()
    pay_cols = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    df['times_late'] = (df[pay_cols] > 0).sum(axis=1)
    df['max_months_late'] = df[pay_cols].max(axis=1)
    
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    df['avg_bill'] = df[bill_cols].mean(axis=1)
    
    amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    df['avg_payment'] = df[amt_cols].mean(axis=1)
    df['payment_ratio'] = df['avg_payment'] / (df['avg_bill'].abs() + 1)
    df['credit_util'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)
    return df

def main():
    print("Loading data...")
    df = load_data()
    
    print("Splitting data...")
    X = df.drop(columns=['default'])
    y = df['default']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Adding features...")
    X_train = add_features(X_train)
    X_test = add_features(X_test)
    
    # Save processed data
    os.makedirs("results", exist_ok=True)
    X_train.to_csv("results/X_train.csv", index=False)
    X_test.to_csv("results/X_test.csv", index=False)
    y_train.to_csv("results/y_train.csv", index=False)
    y_test.to_csv("results/y_test.csv", index=False)
    
    print(f"✓ Saved: X_train ({X_train.shape}), X_test ({X_test.shape})")

if __name__ == "__main__":
    main()

