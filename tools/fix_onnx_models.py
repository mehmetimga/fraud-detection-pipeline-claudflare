#!/usr/bin/env python3
"""
Fix ONNX models to output probabilities as tensors instead of seq(map).
"""

import onnx
from onnx import helper, TensorProto
import numpy as np
import onnxruntime as ort
from pathlib import Path
import shutil

def add_zipmap_extractor(model_path: str, output_path: str, class_idx: int = 1):
    """
    Add post-processing to extract probability tensor from ZipMap output.
    This approach modifies the graph to add extraction logic.
    """
    model = onnx.load(model_path)
    
    # Get the output that produces the ZipMap (probabilities)
    prob_output = None
    for output in model.graph.output:
        if 'prob' in output.name.lower():
            prob_output = output
            break
    
    if prob_output is None:
        print(f"Could not find probability output in {model_path}")
        return False
    
    print(f"Original output: {prob_output.name}, type: {prob_output.type}")
    
    # This is complex - instead, let's use a different approach:
    # Run the model through onnxruntime to get the output, then create a new simpler model
    return False

def convert_catboost_direct(n_features: int, output_path: str):
    """
    Re-export CatBoost with direct probability output.
    CatBoost's native ONNX export doesn't support disabling ZipMap.
    We'll create a wrapper that extracts the probabilities.
    """
    import joblib
    from pathlib import Path
    
    # Load the original CatBoost model
    catboost_model_path = Path("models/catboost_model.cbm")
    if not catboost_model_path.exists():
        # Try to load from training output
        print("CatBoost model file not found. Skipping.")
        return False
    
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(catboost_model_path))
    
    # Export with different parameters
    model.save_model(
        output_path,
        format="onnx",
        export_parameters={
            'onnx_domain': 'ai.catboost',
            'onnx_model_version': 1,
        }
    )
    return True

def create_probability_extraction_wrapper():
    """
    The cleanest solution: Use the 'label' output for classification
    and compute probabilities differently in Rust.
    
    OR: Re-export with skl2onnx which has zipmap: False option.
    """
    pass

def test_existing_models():
    """Test if we can get probabilities from existing models."""
    import pandas as pd
    
    # Load test data
    test_df = pd.read_parquet("training/data/test.parquet")
    
    # Prepare features
    df = test_df.head(5).copy()
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
    y_test = df['is_default'].values
    
    print("\n" + "="*70)
    print("EXTRACTING PROBABILITIES FROM SEQ(MAP) FORMAT")
    print("="*70)
    
    # CatBoost
    print("\n📊 CatBoost:")
    session = ort.InferenceSession("models/catboost.onnx")
    outputs = session.run(None, {"features": X_test})
    probs = outputs[1]  # List of dicts
    
    # Extract class 1 probability from each dict
    catboost_probs = [p[1] for p in probs]
    print(f"   Fraud probabilities: {catboost_probs}")
    print(f"   Actual labels:       {y_test.tolist()}")
    
    # LightGBM
    print("\n📊 LightGBM:")
    session = ort.InferenceSession("models/lightgbm.onnx")
    outputs = session.run(None, {"float_input": X_test})
    probs = outputs[1]  # List of dicts
    
    lightgbm_probs = [p[1] for p in probs]
    print(f"   Fraud probabilities: {lightgbm_probs}")
    print(f"   Actual labels:       {y_test.tolist()}")
    
    # XGBoost (regular tensor)
    print("\n📊 XGBoost:")
    session = ort.InferenceSession("models/xgboost.onnx")
    outputs = session.run(None, {"float_input": X_test})
    xgboost_probs = outputs[1][:, 1].tolist()  # Column 1 is fraud prob
    print(f"   Fraud probabilities: {xgboost_probs}")
    
    # Random Forest (regular tensor)
    print("\n📊 Random Forest:")
    session = ort.InferenceSession("models/random_forest.onnx")
    outputs = session.run(None, {"float_input": X_test})
    rf_probs = outputs[1][:, 1].tolist()
    print(f"   Fraud probabilities: {rf_probs}")
    
    print("\n" + "="*70)
    print("SOLUTION: Need to fix Rust code to extract from seq(map) format")
    print("         OR re-export models without ZipMap")
    print("="*70)

def re_export_lightgbm_without_zipmap(n_features: int = 36):
    """Re-export LightGBM model without zipmap."""
    from onnxmltools.convert import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    import joblib
    
    # Load the trained model
    model_path = Path("models/lightgbm_model.joblib")
    if not model_path.exists():
        print(f"LightGBM model not found at {model_path}")
        return False
    
    model = joblib.load(model_path)
    
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    # Convert with zipmap disabled
    onnx_model = convert_lightgbm(
        model,
        initial_types=initial_type,
        target_opset=12,
        options={'zipmap': False}  # Disable ZipMap!
    )
    
    output_path = "models/lightgbm.onnx"
    onnx.save_model(onnx_model, output_path)
    print(f"✅ LightGBM re-exported to {output_path} (without zipmap)")
    
    # Verify
    session = ort.InferenceSession(output_path)
    for out in session.get_outputs():
        print(f"   Output: {out.name}, shape: {out.shape}, type: {out.type}")
    
    return True

def re_export_catboost_via_sklearn_wrapper(n_features: int = 36):
    """
    CatBoost's native ONNX doesn't support disabling zipmap.
    Alternative: wrap it in a sklearn-compatible interface.
    """
    print("\n⚠️  CatBoost ONNX export doesn't support disabling zipmap natively.")
    print("   Options:")
    print("   1. Use a custom Rust extractor for seq(map)")
    print("   2. Retrain as XGBoost or LightGBM wrapper")
    print("   3. Post-process ONNX graph to add extraction node")
    return False

if __name__ == "__main__":
    print("Testing existing models to extract probabilities...")
    test_existing_models()
    
    print("\n" + "="*70)
    print("RE-EXPORTING MODELS WITHOUT ZIPMAP")
    print("="*70)
    
    # Try to re-export LightGBM
    try:
        re_export_lightgbm_without_zipmap()
    except Exception as e:
        print(f"❌ Failed to re-export LightGBM: {e}")
    
    # CatBoost is more complex
    re_export_catboost_via_sklearn_wrapper()

