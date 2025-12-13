"""
Feature engineering for credit card default detection.
"""

import pandas as pd
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering transformations to credit card data."""
    logger.info("Engineering features...")
    df = df.copy()

    # Payment behavior features
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    if all(col in df.columns for col in pay_cols):
        # Average payment delay
        df['avg_pay_delay'] = df[pay_cols].mean(axis=1)
        
        # Max payment delay
        df['max_pay_delay'] = df[pay_cols].max(axis=1)
        
        # Number of months with delay
        df['months_delayed'] = (df[pay_cols] > 0).sum(axis=1)
        
        # Payment trend (improving or worsening)
        df['pay_trend'] = df['PAY_0'] - df['PAY_6']

    # Bill amount features
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    if all(col in df.columns for col in bill_cols):
        # Average bill amount
        df['avg_bill_amt'] = df[bill_cols].mean(axis=1)
        
        # Bill amount trend
        df['bill_trend'] = df['BILL_AMT1'] - df['BILL_AMT6']
        
        # Bill volatility
        df['bill_volatility'] = df[bill_cols].std(axis=1)

    # Payment amount features  
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    if all(col in df.columns for col in pay_amt_cols):
        # Average payment amount
        df['avg_pay_amt'] = df[pay_amt_cols].mean(axis=1)
        
        # Total payment amount
        df['total_pay_amt'] = df[pay_amt_cols].sum(axis=1)

    # Utilization ratio
    if 'LIMIT_BAL' in df.columns and 'BILL_AMT1' in df.columns:
        df['utilization_ratio'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)
        
        # Over limit indicator
        df['over_limit'] = (df['BILL_AMT1'] > df['LIMIT_BAL']).astype(int)

    # Payment to bill ratio
    if 'PAY_AMT1' in df.columns and 'BILL_AMT1' in df.columns:
        df['pay_bill_ratio'] = df['PAY_AMT1'] / (df['BILL_AMT1'].abs() + 1)

    # Age groups
    if 'AGE' in df.columns:
        df['age_group'] = pd.cut(
            df['AGE'], 
            bins=[0, 25, 35, 45, 55, 100],
            labels=[0, 1, 2, 3, 4]
        ).astype(float).fillna(0)

    logger.info(f"Engineered features, total columns: {len(df.columns)}")
    return df


def get_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """Extract feature importance from a trained model."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'):
        importance = model.get_feature_importance()
    else:
        logger.warning("Model doesn't have feature importance attribute")
        return pd.DataFrame()

    # Handle case where feature names length doesn't match
    if len(feature_names) != len(importance):
        feature_names = [f'feature_{i}' for i in range(len(importance))]

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    df = df.sort_values('importance', ascending=False).head(top_n)
    return df
