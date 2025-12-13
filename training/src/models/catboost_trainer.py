"""
CatBoost model trainer for fraud detection.
"""

import numpy as np
from catboost import CatBoostClassifier
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CatBoostTrainer:
    """Trainer for CatBoost fraud detection model."""

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.1,
        depth: int = 6,
        random_state: int = 42,
        **kwargs
    ):
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=random_state,
            loss_function='Logloss',
            eval_metric='AUC',
            verbose=100,
            auto_class_weights='Balanced',
            **kwargs
        )
        self._fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None
    ) -> 'CatBoostTrainer':
        """Train the CatBoost model."""
        logger.info("Training CatBoost model...")

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)

        # Create Pool for better handling
        from catboost import Pool
        train_pool = Pool(X_train, y_train, feature_names=feature_names)
        eval_pool = Pool(X_val, y_val, feature_names=feature_names) if eval_set else None
        
        self.model.fit(
            train_pool,
            eval_set=eval_pool,
            early_stopping_rounds=50
        )

        self._fitted = True
        logger.info("CatBoost model training complete")
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

    def get_model(self) -> CatBoostClassifier:
        """Get the underlying CatBoost model."""
        return self.model

    def save(self, filepath: str):
        """Save model to file."""
        self.model.save_model(filepath)
        logger.info(f"CatBoost model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'CatBoostTrainer':
        """Load model from file."""
        trainer = cls()
        trainer.model.load_model(filepath)
        trainer._fitted = True
        return trainer

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()

