"""
Random Forest model trainer for fraud detection.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestTrainer:
    """Trainer for Random Forest fraud detection model."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        **kwargs
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1,
            verbose=1,
            **kwargs
        )
        self._fitted = False
        self.feature_names = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None
    ) -> 'RandomForestTrainer':
        """Train the Random Forest model."""
        logger.info("Training Random Forest model...")

        self.feature_names = feature_names
        self.model.fit(X_train, y_train)

        self._fitted = True
        logger.info("Random Forest model training complete")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get binary predictions."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.model.feature_importances_

    def get_model(self) -> RandomForestClassifier:
        """Get the underlying Random Forest model."""
        return self.model

    def save(self, filepath: str):
        """Save model to file."""
        joblib.dump(self.model, filepath)
        logger.info(f"Random Forest model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'RandomForestTrainer':
        """Load model from file."""
        trainer = cls()
        trainer.model = joblib.load(filepath)
        trainer._fitted = True
        return trainer

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()

