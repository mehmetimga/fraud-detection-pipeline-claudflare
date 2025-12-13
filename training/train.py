#!/usr/bin/env python3
"""
Main training script for credit card default detection models.

This script:
1. Loads training data from Parquet files
2. Applies feature engineering
3. Trains all 5 ML models (CatBoost, XGBoost, LightGBM, Random Forest, Isolation Forest)
4. Evaluates and compares model performance
5. Exports models to ONNX format for Rust inference
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import CreditCardPreprocessor, load_parquet_data, get_train_test_data
from src.feature_engineering import engineer_features
from src.evaluation import evaluate_model, compare_models, get_classification_report
from src.export_onnx import export_all_models
from src.models import (
    CatBoostTrainer,
    XGBoostTrainer,
    LightGBMTrainer,
    RandomForestTrainer,
    IsolationForestTrainer
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list
) -> dict:
    """Train all credit card default detection models."""
    models = {}

    # 1. CatBoost
    logger.info("\n" + "=" * 60)
    logger.info("Training CatBoost Model")
    logger.info("=" * 60)
    catboost_trainer = CatBoostTrainer(iterations=300, learning_rate=0.1, depth=6)
    catboost_trainer.fit(X_train, y_train, X_val, y_val, feature_names)
    models['catboost'] = (catboost_trainer, 'catboost')

    # 2. XGBoost
    logger.info("\n" + "=" * 60)
    logger.info("Training XGBoost Model")
    logger.info("=" * 60)
    xgboost_trainer = XGBoostTrainer(n_estimators=300, learning_rate=0.1, max_depth=6)
    xgboost_trainer.fit(X_train, y_train, X_val, y_val, feature_names)
    models['xgboost'] = (xgboost_trainer, 'xgboost')

    # 3. LightGBM
    logger.info("\n" + "=" * 60)
    logger.info("Training LightGBM Model")
    logger.info("=" * 60)
    lightgbm_trainer = LightGBMTrainer(n_estimators=300, learning_rate=0.1, max_depth=6)
    lightgbm_trainer.fit(X_train, y_train, X_val, y_val, feature_names)
    models['lightgbm'] = (lightgbm_trainer, 'lightgbm')

    # 4. Random Forest
    logger.info("\n" + "=" * 60)
    logger.info("Training Random Forest Model")
    logger.info("=" * 60)
    rf_trainer = RandomForestTrainer(n_estimators=200, max_depth=10)
    rf_trainer.fit(X_train, y_train, feature_names=feature_names)
    models['random_forest'] = (rf_trainer, 'random_forest')

    # 5. Isolation Forest
    logger.info("\n" + "=" * 60)
    logger.info("Training Isolation Forest Model")
    logger.info("=" * 60)
    # Set contamination based on actual default rate
    default_rate = y_train.mean()
    iso_trainer = IsolationForestTrainer(n_estimators=200, contamination=default_rate)
    iso_trainer.fit(X_train, y_train, feature_names=feature_names)
    models['isolation_forest'] = (iso_trainer, 'isolation_forest')

    return models


def evaluate_all_models(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """Evaluate all trained models."""
    results = []

    for name, (trainer, _) in models.items():
        y_proba = trainer.predict_proba(X_test)
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = evaluate_model(y_test, y_pred, y_proba, name)
        results.append(metrics)

        # Print classification report
        print(get_classification_report(y_test, y_pred, name))

    # Compare models
    comparison_df = compare_models(results)
    return comparison_df


def main():
    parser = argparse.ArgumentParser(description='Train credit card default detection models')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing parquet files')
    parser.add_argument('--output-dir', type=str, default='../models', help='Output directory for models')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation set size from training data')
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    data_dir = (script_dir / args.data_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CREDIT CARD DEFAULT DETECTION - MODEL TRAINING")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Load data from parquet files
    logger.info(f"\nLoading data from {data_dir}")
    train_df = load_parquet_data(data_dir / 'train.parquet')
    test_df = load_parquet_data(data_dir / 'test.parquet')

    logger.info(f"Training data: {len(train_df)} records")
    logger.info(f"Test data: {len(test_df)} records")
    logger.info(f"Default rate (train): {train_df['is_default'].mean():.2%}")
    logger.info(f"Default rate (test): {test_df['is_default'].mean():.2%}")

    # Apply feature engineering
    logger.info("\nApplying feature engineering...")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    # Preprocess data
    logger.info("\nPreprocessing data...")
    preprocessor = CreditCardPreprocessor()
    
    # Add engineered features to the numeric features list
    engineered_numeric = [
        'avg_pay_delay', 'max_pay_delay', 'months_delayed', 'pay_trend',
        'avg_bill_amt', 'bill_trend', 'bill_volatility',
        'avg_pay_amt', 'total_pay_amt',
        'utilization_ratio', 'over_limit', 'pay_bill_ratio', 'age_group'
    ]
    for feat in engineered_numeric:
        if feat in train_df.columns and feat not in preprocessor.NUMERIC_FEATURES:
            preprocessor.NUMERIC_FEATURES.append(feat)

    X_train_full = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    y_train_full = train_df['is_default'].values
    y_test = test_df['is_default'].values
    feature_names = preprocessor.get_feature_names()

    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=args.val_size,
        random_state=42,
        stratify=y_train_full
    )

    logger.info(f"\nData splits:")
    logger.info(f"  Training: {len(X_train)} samples ({y_train.sum()} defaults)")
    logger.info(f"  Validation: {len(X_val)} samples ({y_val.sum()} defaults)")
    logger.info(f"  Test: {len(X_test)} samples ({y_test.sum()} defaults)")
    logger.info(f"  Features: {len(feature_names)}")

    # Train models
    models = train_all_models(X_train, y_train, X_val, y_val, feature_names)

    # Evaluate models
    logger.info("\n" + "=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)
    comparison_df = evaluate_all_models(models, X_test, y_test)

    # Save comparison results
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    logger.info(f"\nModel comparison saved to {output_dir / 'model_comparison.csv'}")

    # Export to ONNX
    logger.info("\n" + "=" * 60)
    logger.info("EXPORTING MODELS TO ONNX")
    logger.info("=" * 60)
    export_all_models(models, str(output_dir), len(feature_names), feature_names)

    # Save feature names and preprocessing info for Rust inference
    feature_info = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'dataset': 'UCI Credit Card Default',
        'target': 'is_default',
        'preprocessing': {
            'numeric_features': preprocessor.NUMERIC_FEATURES,
            'categorical_features': preprocessor.CATEGORICAL_FEATURES,
        },
        'training_info': {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'default_rate': float(y_train.mean()),
        }
    }
    with open(output_dir / 'feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    logger.info(f"Feature info saved to {output_dir / 'feature_info.json'}")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Models saved to: {output_dir}")
    logger.info("=" * 60)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nBest model by ROC-AUC: {comparison_df.iloc[0]['model']}")
    print(f"  ROC-AUC: {comparison_df.iloc[0]['roc_auc']:.4f}")
    print(f"  PR-AUC: {comparison_df.iloc[0]['pr_auc']:.4f}")
    print(f"  F1: {comparison_df.iloc[0]['f1']:.4f}")
    print("\nAll models exported to ONNX format for Rust inference.")


if __name__ == '__main__':
    main()
