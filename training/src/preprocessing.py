"""
Data preprocessing for credit card default detection training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditCardPreprocessor:
    """Preprocessor for credit card default detection data."""

    # Feature columns for the UCI Credit Card Default dataset
    NUMERIC_FEATURES = [
        'LIMIT_BAL',      # Credit limit
        'AGE',            # Age
        'PAY_0',          # Payment status month 0
        'PAY_2',          # Payment status month 2
        'PAY_3',          # Payment status month 3
        'PAY_4',          # Payment status month 4
        'PAY_5',          # Payment status month 5
        'PAY_6',          # Payment status month 6
        'BILL_AMT1',      # Bill amount month 1
        'BILL_AMT2',      # Bill amount month 2
        'BILL_AMT3',      # Bill amount month 3
        'BILL_AMT4',      # Bill amount month 4
        'BILL_AMT5',      # Bill amount month 5
        'BILL_AMT6',      # Bill amount month 6
        'PAY_AMT1',       # Payment amount month 1
        'PAY_AMT2',       # Payment amount month 2
        'PAY_AMT3',       # Payment amount month 3
        'PAY_AMT4',       # Payment amount month 4
        'PAY_AMT5',       # Payment amount month 5
        'PAY_AMT6',       # Payment amount month 6
    ]

    CATEGORICAL_FEATURES = [
        'SEX',            # 1 = male, 2 = female
        'EDUCATION',      # 1 = graduate, 2 = university, 3 = high school, 4 = others
        'MARRIAGE',       # 1 = married, 2 = single, 3 = others
    ]

    TARGET_COLUMN = 'is_default'

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> 'CreditCardPreprocessor':
        """Fit the preprocessor on training data."""
        logger.info("Fitting preprocessor on training data...")

        # Fit scaler on numeric features
        numeric_cols = [c for c in self.NUMERIC_FEATURES if c in df.columns]
        if numeric_cols:
            self.scaler.fit(df[numeric_cols].fillna(0))

        # Build feature names list
        self.feature_names = []
        self.feature_names.extend(numeric_cols)
        self.feature_names.extend([c for c in self.CATEGORICAL_FEATURES if c in df.columns])

        self._fitted = True
        logger.info(f"Preprocessor fitted with {len(self.feature_names)} features")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if not self._fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        features = []

        # Scale numeric features
        numeric_cols = [c for c in self.NUMERIC_FEATURES if c in df.columns]
        if numeric_cols:
            numeric_data = df[numeric_cols].fillna(0)
            scaled = self.scaler.transform(numeric_data)
            features.append(scaled)

        # Categorical features as-is (they're already encoded as integers)
        for col in self.CATEGORICAL_FEATURES:
            if col in df.columns:
                cat_data = df[col].fillna(0).astype(float).values.reshape(-1, 1)
                features.append(cat_data)

        return np.hstack(features).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    def get_feature_names(self) -> List[str]:
        """Get the list of feature names."""
        return self.feature_names.copy()


def load_parquet_data(filepath: str) -> pd.DataFrame:
    """Load data from Parquet file."""
    logger.info(f"Loading data from {filepath}...")
    df = pd.read_parquet(filepath)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file."""
    logger.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} records")
    return df


def get_train_test_data(
    data_dir: str = 'data'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-split train and test data from parquet files."""
    import os
    
    train_path = os.path.join(data_dir, 'train.parquet')
    test_path = os.path.join(data_dir, 'test.parquet')
    
    train_df = load_parquet_data(train_path)
    test_df = load_parquet_data(test_path)
    
    return train_df, test_df


def prepare_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'is_default'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare training and test data.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    preprocessor = CreditCardPreprocessor()
    
    # Get features and target
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    
    feature_names = preprocessor.get_feature_names()
    
    logger.info(f"Prepared {len(X_train)} training samples, {len(X_test)} test samples")
    logger.info(f"Feature count: {len(feature_names)}")
    
    return X_train, X_test, y_train, y_test, feature_names
