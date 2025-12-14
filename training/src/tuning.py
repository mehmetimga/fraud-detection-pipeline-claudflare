"""
Hyperparameter tuning and threshold optimization for fraud detection models.
"""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve, make_scorer
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Hyperparameter search spaces
CATBOOST_PARAM_DIST = {
    'iterations': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'depth': [4, 5, 6, 7, 8],
    'l2_leaf_reg': [1, 3, 5, 7],
}

XGBOOST_PARAM_DIST = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
}

LIGHTGBM_PARAM_DIST = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7],
    'num_leaves': [15, 31, 63, 127],
    'min_child_samples': [10, 20, 30, 50],
    'subsample': [0.7, 0.8, 0.9, 1.0],
}

RANDOM_FOREST_PARAM_DIST = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 8, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
}


def tune_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 42
) -> Tuple[CatBoostClassifier, Dict[str, Any], float]:
    """Tune CatBoost hyperparameters."""
    logger.info("Tuning CatBoost hyperparameters...")
    
    base_model = CatBoostClassifier(
        random_seed=random_state,
        loss_function='Logloss',
        eval_metric='AUC',
        verbose=0,
        auto_class_weights='Balanced',
    )
    
    search = RandomizedSearchCV(
        base_model,
        CATBOOST_PARAM_DIST,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring='f1',
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    logger.info(f"Best CatBoost F1: {search.best_score_:.4f}")
    logger.info(f"Best CatBoost params: {search.best_params_}")
    
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 42
) -> Tuple[xgb.XGBClassifier, Dict[str, Any], float]:
    """Tune XGBoost hyperparameters."""
    logger.info("Tuning XGBoost hyperparameters...")
    
    # Calculate scale_pos_weight
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
    
    base_model = xgb.XGBClassifier(
        random_state=random_state,
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
    )
    
    search = RandomizedSearchCV(
        base_model,
        XGBOOST_PARAM_DIST,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring='f1',
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    logger.info(f"Best XGBoost F1: {search.best_score_:.4f}")
    logger.info(f"Best XGBoost params: {search.best_params_}")
    
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 42
) -> Tuple[lgb.LGBMClassifier, Dict[str, Any], float]:
    """Tune LightGBM hyperparameters."""
    logger.info("Tuning LightGBM hyperparameters...")
    
    base_model = lgb.LGBMClassifier(
        random_state=random_state,
        objective='binary',
        metric='auc',
        is_unbalance=True,
        verbosity=-1,
    )
    
    search = RandomizedSearchCV(
        base_model,
        LIGHTGBM_PARAM_DIST,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring='f1',
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    logger.info(f"Best LightGBM F1: {search.best_score_:.4f}")
    logger.info(f"Best LightGBM params: {search.best_params_}")
    
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, Dict[str, Any], float]:
    """Tune Random Forest hyperparameters."""
    logger.info("Tuning Random Forest hyperparameters...")
    
    base_model = RandomForestClassifier(
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1,
    )
    
    search = RandomizedSearchCV(
        base_model,
        RANDOM_FOREST_PARAM_DIST,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring='f1',
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    logger.info(f"Best Random Forest F1: {search.best_score_:.4f}")
    logger.info(f"Best Random Forest params: {search.best_params_}")
    
    return search.best_estimator_, search.best_params_, search.best_score_


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find the optimal classification threshold for a given metric.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')
    
    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Remove last element (precision/recall at threshold=1)
    precision = precision[:-1]
    recall = recall[:-1]
    
    if metric == 'f1':
        # F1 = 2 * (precision * recall) / (precision + recall)
        scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    elif metric == 'precision':
        scores = precision
    elif metric == 'recall':
        scores = recall
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    best_idx = np.argmax(scores)
    optimal_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    logger.info(f"Optimal threshold for {metric}: {optimal_threshold:.4f} (score: {best_score:.4f})")
    
    return optimal_threshold, best_score


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """Evaluate predictions at a specific threshold."""
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    y_pred = (y_proba >= threshold).astype(int)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

