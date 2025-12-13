"""
Model evaluation and comparison metrics for fraud detection.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str
) -> Dict[str, float]:
    """
    Evaluate a single model's performance.
    
    Args:
        y_true: True labels
        y_pred: Binary predictions
        y_proba: Probability predictions
        model_name: Name of the model
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'pr_auc': average_precision_score(y_true, y_proba),
    }

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)

    # Additional fraud-specific metrics
    metrics['fraud_detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0

    logger.info(f"\n{model_name} Evaluation:")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1']:.4f}")
    logger.info(f"  Fraud Detection Rate: {metrics['fraud_detection_rate']:.4f}")

    return metrics


def compare_models(
    results: List[Dict[str, float]],
    primary_metric: str = 'roc_auc'
) -> pd.DataFrame:
    """
    Compare multiple models and rank them.
    
    Args:
        results: List of evaluation results from evaluate_model
        primary_metric: Metric to use for ranking
        
    Returns:
        DataFrame with model comparison
    """
    df = pd.DataFrame(results)
    df = df.sort_values(primary_metric, ascending=False)
    df['rank'] = range(1, len(df) + 1)

    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON (Ranked by {})".format(primary_metric))
    logger.info("=" * 60)

    for _, row in df.iterrows():
        logger.info(f"\n#{int(row['rank'])} {row['model']}")
        logger.info(f"   ROC-AUC: {row['roc_auc']:.4f}")
        logger.info(f"   PR-AUC: {row['pr_auc']:.4f}")
        logger.info(f"   F1: {row['f1']:.4f}")

    return df


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str
) -> str:
    """Get detailed classification report."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=['Legitimate', 'Fraud'],
        digits=4
    )
    return f"\n{model_name} Classification Report:\n{report}"


def calculate_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray,
    model_name: str
) -> Dict[str, float]:
    """
    Calculate business-relevant metrics for fraud detection.
    
    Args:
        y_true: True labels
        y_pred: Binary predictions
        amounts: Transaction amounts
        model_name: Name of the model
        
    Returns:
        Dictionary of business metrics
    """
    # Calculate fraud amount metrics
    fraud_mask = y_true == 1
    detected_fraud_mask = (y_true == 1) & (y_pred == 1)
    missed_fraud_mask = (y_true == 1) & (y_pred == 0)
    false_alarm_mask = (y_true == 0) & (y_pred == 1)

    total_fraud_amount = amounts[fraud_mask].sum()
    detected_fraud_amount = amounts[detected_fraud_mask].sum()
    missed_fraud_amount = amounts[missed_fraud_mask].sum()
    false_alarm_amount = amounts[false_alarm_mask].sum()

    metrics = {
        'model': model_name,
        'total_fraud_amount': float(total_fraud_amount),
        'detected_fraud_amount': float(detected_fraud_amount),
        'missed_fraud_amount': float(missed_fraud_amount),
        'false_alarm_amount': float(false_alarm_amount),
        'fraud_amount_detection_rate': (
            detected_fraud_amount / total_fraud_amount 
            if total_fraud_amount > 0 else 0
        ),
        'avg_detected_fraud_amount': (
            detected_fraud_amount / detected_fraud_mask.sum()
            if detected_fraud_mask.sum() > 0 else 0
        ),
        'avg_missed_fraud_amount': (
            missed_fraud_amount / missed_fraud_mask.sum()
            if missed_fraud_mask.sum() > 0 else 0
        ),
    }

    logger.info(f"\n{model_name} Business Metrics:")
    logger.info(f"  Total Fraud Amount: ${total_fraud_amount:,.2f}")
    logger.info(f"  Detected Fraud Amount: ${detected_fraud_amount:,.2f}")
    logger.info(f"  Missed Fraud Amount: ${missed_fraud_amount:,.2f}")
    logger.info(f"  Fraud Amount Detection Rate: {metrics['fraud_amount_detection_rate']:.2%}")

    return metrics

