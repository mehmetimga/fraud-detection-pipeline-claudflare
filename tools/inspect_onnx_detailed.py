#!/usr/bin/env python3
"""
Detailed inspection of CatBoost/LightGBM ONNX output format.
"""

import onnx
import numpy as np
import onnxruntime as ort
import pandas as pd
from pathlib import Path

def main():
    # Load test data
    test_df = pd.read_parquet("training/data/test.parquet")
    
    # Prepare features (same as before)
    df = test_df.head(3).copy()
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    df['avg_pay_delay'] = df[pay_cols].mean(axis=1)
    df['max_pay_delay'] = df[pay_cols].max(axis=1)
    df['months_delayed'] = (df[pay_cols] > 0).sum(axis=1)
    df['pay_trend'] = df['PAY_0'] - df['PAY_6']
    df['avg_bill_amt'] = df[bill_cols].mean(axis=1)
    df['bill_trend'] = df['BILL_AMT1'] - df['BILL_AMT6']
    df['bill_volatility'] = df[bill_cols].std(axis=1)
    df['avg_pay_amt'] = df[pay_amt_cols].mean(axis=1)
    df['total_pay_amt'] = df[pay_amt_cols].sum(axis=1)
    df['utilization_ratio'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)
    df['over_limit'] = (df['BILL_AMT1'] > df['LIMIT_BAL']).astype(int)
    df['pay_bill_ratio'] = df['PAY_AMT1'] / (df['BILL_AMT1'].abs() + 1)
    df['age_group'] = pd.cut(df['AGE'], bins=[0, 25, 35, 45, 55, 100], labels=[0, 1, 2, 3, 4]).astype(int)
    
    feature_cols = [
        'LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    all_features = feature_cols + [
        'avg_pay_delay', 'max_pay_delay', 'months_delayed', 'pay_trend',
        'avg_bill_amt', 'bill_trend', 'bill_volatility', 'avg_pay_amt',
        'total_pay_amt', 'utilization_ratio', 'over_limit', 'pay_bill_ratio',
        'age_group', 'SEX', 'EDUCATION', 'MARRIAGE'
    ]
    
    X_test = df[all_features].values.astype(np.float32)
    
    # Inspect CatBoost
    print("\n" + "="*70)
    print("CATBOOST DETAILED INSPECTION")
    print("="*70)
    
    session = ort.InferenceSession("models/catboost.onnx")
    print("\nOutputs info:")
    for output in session.get_outputs():
        print(f"  Name: {output.name}")
        print(f"  Shape: {output.shape}")
        print(f"  Type: {output.type}")
        print()
    
    outputs = session.run(None, {"features": X_test})
    print("Raw output values:")
    for i, out in enumerate(outputs):
        name = session.get_outputs()[i].name
        print(f"\n  Output[{i}] '{name}':")
        print(f"    Python type: {type(out)}")
        if isinstance(out, list):
            print(f"    Length: {len(out)}")
            print(f"    First element type: {type(out[0])}")
            print(f"    First element: {out[0]}")
            if isinstance(out[0], dict):
                print(f"    Dict keys: {out[0].keys()}")
        else:
            print(f"    Value: {out}")
    
    # Inspect LightGBM
    print("\n" + "="*70)
    print("LIGHTGBM DETAILED INSPECTION")
    print("="*70)
    
    session = ort.InferenceSession("models/lightgbm.onnx")
    print("\nOutputs info:")
    for output in session.get_outputs():
        print(f"  Name: {output.name}")
        print(f"  Shape: {output.shape}")
        print(f"  Type: {output.type}")
        print()
    
    outputs = session.run(None, {"float_input": X_test})
    print("Raw output values:")
    for i, out in enumerate(outputs):
        name = session.get_outputs()[i].name
        print(f"\n  Output[{i}] '{name}':")
        print(f"    Python type: {type(out)}")
        if isinstance(out, list):
            print(f"    Length: {len(out)}")
            print(f"    First element type: {type(out[0])}")
            print(f"    First element: {out[0]}")
            if isinstance(out[0], dict):
                print(f"    Dict keys: {out[0].keys()}")
        else:
            print(f"    Value: {out}")

if __name__ == "__main__":
    main()

