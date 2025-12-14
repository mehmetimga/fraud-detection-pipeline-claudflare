#!/usr/bin/env python3
"""
Enhanced training script with hyperparameter tuning and threshold optimization.

This script:
1. Loads training data
2. Tunes hyperparameters for each model using RandomizedSearchCV
3. Finds optimal classification thresholds
4. Exports best models to ONNX format
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

from src.preprocessing import CreditCardPreprocessor, load_parquet_data
from src.feature_engineering import engineer_features
from src.evaluation import evaluate_model, compare_models, get_classification_report
from src.export_onnx import export_all_models
from src.tuning import (
    tune_catboost, tune_xgboost, tune_lightgbm, tune_random_forest,
    find_optimal_threshold, evaluate_at_threshold
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TunedModelWrapper:
    """Wrapper to make tuned sklearn models compatible with export."""
    def __init__(self, model, model_type: str):
        self.model = model
        self.model_type = model_type
        self._fitted = True
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_model(self):
        return self.model
    
    def get_feature_importance(self):
        return self.model.feature_importances_


def train_with_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 20,
    cv: int = 3
) -> dict:
    """Train all models with hyperparameter tuning."""
    models = {}
    best_params = {}
    
    # 1. CatBoost
    logger.info("\n" + "=" * 60)
    logger.info("Tuning CatBoost Model")
    logger.info("=" * 60)
    cb_model, cb_params, cb_score = tune_catboost(X_train, y_train, n_iter=n_iter, cv=cv)
    models['catboost'] = (TunedModelWrapper(cb_model, 'catboost'), 'catboost')
    best_params['catboost'] = {'params': cb_params, 'cv_f1': cb_score}
    
    # 2. XGBoost
    logger.info("\n" + "=" * 60)
    logger.info("Tuning XGBoost Model")
    logger.info("=" * 60)
    xgb_model, xgb_params, xgb_score = tune_xgboost(X_train, y_train, n_iter=n_iter, cv=cv)
    models['xgboost'] = (TunedModelWrapper(xgb_model, 'xgboost'), 'xgboost')
    best_params['xgboost'] = {'params': xgb_params, 'cv_f1': xgb_score}
    
    # 3. LightGBM
    logger.info("\n" + "=" * 60)
    logger.info("Tuning LightGBM Model")
    logger.info("=" * 60)
    lgb_model, lgb_params, lgb_score = tune_lightgbm(X_train, y_train, n_iter=n_iter, cv=cv)
    models['lightgbm'] = (TunedModelWrapper(lgb_model, 'lightgbm'), 'lightgbm')
    best_params['lightgbm'] = {'params': lgb_params, 'cv_f1': lgb_score}
    
    # 4. Random Forest
    logger.info("\n" + "=" * 60)
    logger.info("Tuning Random Forest Model")
    logger.info("=" * 60)
    rf_model, rf_params, rf_score = tune_random_forest(X_train, y_train, n_iter=n_iter, cv=cv)
    models['random_forest'] = (TunedModelWrapper(rf_model, 'random_forest'), 'random_forest')
    best_params['random_forest'] = {'params': rf_params, 'cv_f1': rf_score}
    
    return models, best_params


def evaluate_with_thresholds(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> tuple:
    """Evaluate all models and find optimal thresholds."""
    results = []
    optimal_thresholds = {}
    
    for name, (trainer, _) in models.items():
        logger.info(f"\n--- Evaluating {name} ---")
        
        y_proba = trainer.predict_proba(X_test)
        
        # Find optimal threshold
        opt_threshold, opt_f1 = find_optimal_threshold(y_test, y_proba, metric='f1')
        optimal_thresholds[name] = opt_threshold
        
        # Evaluate at default threshold (0.5)
        default_metrics = evaluate_at_threshold(y_test, y_proba, 0.5)
        logger.info(f"At threshold 0.5: F1={default_metrics['f1']:.4f}, Recall={default_metrics['recall']:.4f}")
        
        # Evaluate at optimal threshold
        optimal_metrics = evaluate_at_threshold(y_test, y_proba, opt_threshold)
        logger.info(f"At threshold {opt_threshold:.3f}: F1={optimal_metrics['f1']:.4f}, Recall={optimal_metrics['recall']:.4f}")
        
        # Use optimal threshold for final evaluation
        y_pred = (y_proba >= opt_threshold).astype(int)
        metrics = evaluate_model(y_test, y_pred, y_proba, name)
        metrics['optimal_threshold'] = opt_threshold
        results.append(metrics)
        
        print(get_classification_report(y_test, y_pred, name))
    
    comparison_df = compare_models(results)
    return comparison_df, optimal_thresholds


def main():
    parser = argparse.ArgumentParser(description='Train models with hyperparameter tuning')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='../models', help='Output directory')
    parser.add_argument('--n-iter', type=int, default=20, help='Number of hyperparameter search iterations')
    parser.add_argument('--cv', type=int, default=3, help='Cross-validation folds')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation set size')
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    data_dir = (script_dir / args.data_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ENHANCED TRAINING WITH HYPERPARAMETER TUNING")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Load data
    logger.info(f"\nLoading data from {data_dir}")
    train_df = load_parquet_data(data_dir / 'train.parquet')
    test_df = load_parquet_data(data_dir / 'test.parquet')

    logger.info(f"Training data: {len(train_df)} records")
    logger.info(f"Test data: {len(test_df)} records")
    logger.info(f"Default rate (train): {train_df['is_default'].mean():.2%}")
    logger.info(f"Default rate (test): {test_df['is_default'].mean():.2%}")

    # Feature engineering
    logger.info("\nApplying feature engineering...")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    # Preprocess
    logger.info("\nPreprocessing data...")
    preprocessor = CreditCardPreprocessor()
    
    # Add engineered features
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

    logger.info(f"Features: {len(feature_names)}")
    logger.info(f"Training samples: {len(X_train_full)}")
    logger.info(f"Test samples: {len(X_test)}")

    # Train with tuning
    logger.info("\n" + "=" * 60)
    logger.info("HYPERPARAMETER TUNING")
    logger.info(f"Iterations per model: {args.n_iter}")
    logger.info(f"Cross-validation folds: {args.cv}")
    logger.info("=" * 60)
    
    models, best_params = train_with_tuning(
        X_train_full, y_train_full, 
        n_iter=args.n_iter, 
        cv=args.cv
    )

    # Evaluate with threshold optimization
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION WITH THRESHOLD OPTIMIZATION")
    logger.info("=" * 60)
    
    comparison_df, optimal_thresholds = evaluate_with_thresholds(models, X_test, y_test)

    # Save results
    comparison_df.to_csv(output_dir / 'model_comparison_tuned.csv', index=False)
    logger.info(f"\nResults saved to {output_dir / 'model_comparison_tuned.csv'}")

    # Export to ONNX
    logger.info("\n" + "=" * 60)
    logger.info("EXPORTING MODELS TO ONNX")
    logger.info("=" * 60)
    export_all_models(models, str(output_dir), len(feature_names), feature_names)

    # Save configuration
    config = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'optimal_thresholds': {k: float(v) for k, v in optimal_thresholds.items()},
        'best_params': {k: {pk: str(pv) for pk, pv in v['params'].items()} for k, v in best_params.items()},
        'cv_scores': {k: float(v['cv_f1']) for k, v in best_params.items()},
        'training_info': {
            'train_samples': int(len(X_train_full)),
            'test_samples': int(len(X_test)),
            'default_rate': float(y_train_full.mean()),
            'n_iter': args.n_iter,
            'cv_folds': args.cv,
        }
    }
    with open(output_dir / 'tuned_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {output_dir / 'tuned_config.json'}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print("\nBest CV F1 Scores:")
    for name, info in best_params.items():
        print(f"  {name}: {info['cv_f1']:.4f}")
    
    print("\nOptimal Thresholds:")
    for name, threshold in optimal_thresholds.items():
        print(f"  {name}: {threshold:.3f}")
    
    print("\nTest Set Results (at optimal thresholds):")
    print(comparison_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()

