"""
Train models for credit card default prediction.
Uses same models as Rust project: XGBoost, CatBoost, LightGBM, Random Forest.

Usage: python src/train.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import joblib
import json
import os
from datetime import datetime


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model with early stopping."""
    print("\n--- Training XGBoost ---")
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc',
        early_stopping_rounds=50,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    
    print(f"✓ XGBoost trained ({model.best_iteration} rounds)")
    return model


def train_catboost(X_train, y_train, X_val, y_val):
    """Train CatBoost model with early stopping."""
    print("\n--- Training CatBoost ---")
    
    model = cb.CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        auto_class_weights='Balanced',
        random_seed=42,
        early_stopping_rounds=50,
        verbose=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    
    print(f"✓ CatBoost trained ({model.best_iteration_} iterations)")
    return model


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM model with early stopping."""
    print("\n--- Training LightGBM ---")
    
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        is_unbalance=True,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )
    
    print(f"✓ LightGBM trained ({model.best_iteration_} rounds)")
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    print("\n--- Training Random Forest ---")
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print(f"✓ Random Forest trained ({model.n_estimators} trees)")
    return model


def validate_model(model, X_val, y_val, model_name):
    """Validate a model and return ROC-AUC."""
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_val)[:, 1]
    else:
        y_proba = model.predict(X_val)
    
    auc = roc_auc_score(y_val, y_proba)
    print(f"{model_name} Validation ROC-AUC: {auc:.4f}")
    return auc


def main():
    print("=" * 60)
    print("MODEL TRAINING")
    print("Models: XGBoost, CatBoost, LightGBM, Random Forest")
    print("=" * 60)
    
    # Load processed data
    print("\nLoading processed data...")
    X_train_full = pd.read_csv("results/X_train.csv")
    y_train_full = pd.read_csv("results/y_train.csv").values.ravel()
    
    print(f"Train samples: {len(X_train_full)}")
    print(f"Default rate: {y_train_full.mean():.1%}")
    print(f"Features: {len(X_train_full.columns)}")
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.1,
        random_state=42,
        stratify=y_train_full
    )
    
    print(f"\nTraining: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    
    # Train all models
    models = {}
    
    # 1. XGBoost
    models['xgboost'] = train_xgboost(X_train, y_train, X_val, y_val)
    
    # 2. CatBoost
    models['catboost'] = train_catboost(X_train, y_train, X_val, y_val)
    
    # 3. LightGBM
    models['lightgbm'] = train_lightgbm(X_train, y_train, X_val, y_val)
    
    # 4. Random Forest
    models['random_forest'] = train_random_forest(X_train, y_train)
    
    # Validate all models
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    val_scores = {}
    for name, model in models.items():
        auc = validate_model(model, X_val, y_val, name)
        val_scores[name] = auc
    
    # Save models
    print("\n--- Saving Models ---")
    os.makedirs("results", exist_ok=True)
    
    for name, model in models.items():
        model_path = f"results/{name}_model.pkl"
        joblib.dump(model, model_path)
        print(f"✓ Saved: {model_path}")
    
    # Save validation scores
    val_df = pd.DataFrame([
        {'model': name, 'val_roc_auc': score}
        for name, score in val_scores.items()
    ]).sort_values('val_roc_auc', ascending=False)
    
    val_df.to_csv("results/validation_scores.csv", index=False)
    
    # Save feature names
    feature_names = list(X_train.columns)
    with open("results/feature_names.json", "w") as f:
        json.dump({
            'features': feature_names,
            'n_features': len(feature_names),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print("\n✓ All models saved to results/")
    print(f"\nBest model: {val_df.iloc[0]['model']} (ROC-AUC: {val_df.iloc[0]['val_roc_auc']:.4f})")


if __name__ == "__main__":
    main()
