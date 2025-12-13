"""
Isolation Forest model trainer for fraud detection (anomaly detection approach).
"""

import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IsolationForestTrainer:
    """
    Trainer for Isolation Forest fraud detection model.
    
    This is an unsupervised anomaly detection model that doesn't use labels.
    It learns the normal pattern and flags anomalies (potential fraud).
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: str = 'auto',
        contamination: float = 0.03,  # Approximate fraud rate
        random_state: int = 42,
        **kwargs
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
            **kwargs
        )
        self._fitted = False
        self.feature_names = None
        self.contamination = contamination

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray = None,  # Not used for unsupervised learning
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None
    ) -> 'IsolationForestTrainer':
        """
        Train the Isolation Forest model.
        
        Note: This is unsupervised, so y_train is ignored.
        The model learns from normal transactions only.
        """
        logger.info("Training Isolation Forest model...")

        self.feature_names = feature_names

        # For better results, train only on non-fraud transactions
        if y_train is not None:
            X_normal = X_train[y_train == 0]
            logger.info(f"Training on {len(X_normal)} normal transactions")
            self.model.fit(X_normal)
        else:
            self.model.fit(X_train)

        self._fitted = True
        logger.info("Isolation Forest model training complete")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores as probability-like values.
        
        Converts isolation forest scores to 0-1 range where:
        - Higher values = more anomalous (potential fraud)
        - Lower values = more normal
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        # Get raw anomaly scores (negative = more anomalous)
        raw_scores = self.model.decision_function(X)

        # Convert to probability-like scores (0-1 range, higher = more fraud-like)
        # Isolation Forest returns negative scores for anomalies
        # We invert and normalize to get fraud probability
        min_score = raw_scores.min()
        max_score = raw_scores.max()

        if max_score > min_score:
            # Normalize to 0-1 and invert (so anomalies get high scores)
            normalized = (raw_scores - min_score) / (max_score - min_score)
            proba = 1 - normalized  # Invert so anomalies get high scores
        else:
            proba = np.zeros(len(X))

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get binary predictions.
        
        Returns 1 for anomalies (potential fraud), 0 for normal.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        # Isolation Forest returns -1 for anomalies, 1 for normal
        predictions = self.model.predict(X)
        # Convert to 0/1 where 1 = fraud
        return (predictions == -1).astype(int)

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.
        
        Note: Isolation Forest doesn't have direct feature importance.
        Returns uniform weights.
        """
        if self.feature_names:
            return np.ones(len(self.feature_names)) / len(self.feature_names)
        return np.array([])

    def get_model(self) -> IsolationForest:
        """Get the underlying Isolation Forest model."""
        return self.model

    def save(self, filepath: str):
        """Save model to file."""
        joblib.dump(self.model, filepath)
        logger.info(f"Isolation Forest model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'IsolationForestTrainer':
        """Load model from file."""
        trainer = cls()
        trainer.model = joblib.load(filepath)
        trainer._fitted = True
        return trainer

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()

