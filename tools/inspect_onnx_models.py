#!/usr/bin/env python3
"""
Inspect ONNX models to understand their input/output format.
"""

import onnx
import numpy as np
import onnxruntime as ort
import pandas as pd
from pathlib import Path
import json

def inspect_model(model_path: str):
    """Inspect ONNX model structure."""
    print(f"\n{'='*60}")
    print(f"MODEL: {Path(model_path).name}")
    print('='*60)
    
    # Load ONNX model for structure info
    model = onnx.load(model_path)
    
    # Inputs
    print("\n📥 INPUTS:")
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else 'batch' for d in inp.type.tensor_type.shape.dim]
        elem_type = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
        print(f"   - {inp.name}: shape={shape}, type={elem_type}")
    
    # Outputs
    print("\n📤 OUTPUTS:")
    for out in model.graph.output:
        if hasattr(out.type, 'tensor_type'):
            shape = [d.dim_value if d.dim_value > 0 else 'batch' for d in out.type.tensor_type.shape.dim]
            elem_type = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
            print(f"   - {out.name}: shape={shape}, type={elem_type}")
        else:
            print(f"   - {out.name}: type=sequence/map")
    
    return model

def run_inference_test(model_path: str, test_data: np.ndarray):
    """Run inference and show outputs."""
    session = ort.InferenceSession(model_path)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    print(f"\n🔧 Input name: {input_name}")
    print(f"   Test data shape: {test_data.shape}")
    
    # Run inference
    outputs = session.run(None, {input_name: test_data})
    
    print(f"\n📊 OUTPUT VALUES ({len(outputs)} outputs):")
    for i, output in enumerate(outputs):
        output_name = session.get_outputs()[i].name
        print(f"\n   Output[{i}] - '{output_name}':")
        print(f"      Shape: {output.shape}")
        print(f"      Type: {output.dtype}")
        if isinstance(output, np.ndarray):
            print(f"      Values (first 3 rows):")
            for row_idx, row in enumerate(output[:3]):
                print(f"         Row {row_idx}: {row}")
        else:
            print(f"      Values: {output[:3]}")
    
    return outputs

def main():
    models_dir = Path("models")
    
    # Load test data
    test_df = pd.read_parquet("training/data/test.parquet")
    
    # Features to use (must match training)
    feature_cols = [
        'LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
        'SEX', 'EDUCATION', 'MARRIAGE'
    ]
    
    # Add engineered features
    df = test_df.copy()
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
    
    # Full feature list (36 features)
    all_features = feature_cols[:20] + [  # First 20 numeric
        'avg_pay_delay', 'max_pay_delay', 'months_delayed', 'pay_trend',
        'avg_bill_amt', 'bill_trend', 'bill_volatility', 'avg_pay_amt',
        'total_pay_amt', 'utilization_ratio', 'over_limit', 'pay_bill_ratio',
        'age_group', 'SEX', 'EDUCATION', 'MARRIAGE'  # Last 3 from original
    ]
    
    # Get 3 test samples (include one default and one non-default)
    test_samples = df.head(3)
    X_test = test_samples[all_features].values.astype(np.float32)
    y_test = test_samples['is_default'].values
    
    print("\n" + "="*60)
    print("TEST SAMPLES")
    print("="*60)
    print(f"Shape: {X_test.shape}")
    print(f"Labels (is_default): {y_test}")
    
    # Inspect each model
    for model_file in sorted(models_dir.glob("*.onnx")):
        try:
            inspect_model(str(model_file))
            outputs = run_inference_test(str(model_file), X_test)
        except Exception as e:
            print(f"\n❌ Error with {model_file.name}: {e}")

if __name__ == "__main__":
    main()

